#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <faiss/Index.h>
#include <faiss/IndexHNSW.h>
#include <faiss/impl/FaissException.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace py = pybind11;

/*
 * TopLoc-HNSW custom level-0 beam search.
 *
 * Why this exists:
 *   FAISS Python does not expose an API to start HNSW search from a custom
 *   entry point. TopLoc-HNSW needs exactly that: q0 selects a privileged entry
 *   point, and follow-up turns start level-0 search from this cached node.
 *
 * Inputs:
 *   index_ptr: raw pointer to a FAISS IndexHNSW / IndexHNSWFlat object
 *   q_emb:     (nq, d) float32 query matrix, or (d,) for one query
 *   entry_points: either (ep,) shared by all queries, or (nq, ep)
 *   offsets, neighbors, degree0: level-0 graph arrays copied from FAISS HNSW
 *   k, ef_search: usual HNSW search parameters
 *
 * Output:
 *   (scores, indices, visited_counts)
 *   scores/indices have shape (nq, k). visited_counts has shape (nq,).
 */

using idx_t = faiss::idx_t;

static inline float dot_product(const float* a, const float* b, int d) {
    float s = 0.0f;
    for (int i = 0; i < d; ++i) s += a[i] * b[i];
    return s;
}

static inline float neg_l2_distance(const float* a, const float* b, int d) {
    float s = 0.0f;
    for (int i = 0; i < d; ++i) {
        float diff = a[i] - b[i];
        s += diff * diff;
    }
    return -s;  // higher is better, like inner product
}

struct CandidateMaxCmp {
    bool operator()(const std::pair<float, idx_t>& a, const std::pair<float, idx_t>& b) const {
        return a.first < b.first; // max heap by score
    }
};

struct ResultMinCmp {
    bool operator()(const std::pair<float, idx_t>& a, const std::pair<float, idx_t>& b) const {
        return a.first > b.first; // min heap by score
    }
};

py::tuple toploc_hnsw_level0_search_ptr(
    uintptr_t index_ptr,
    py::array_t<float, py::array::c_style | py::array::forcecast> q_emb,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> entry_points,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> offsets,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> neighbors,
    int degree0,
    int k,
    int ef_search
) {
    auto* index = reinterpret_cast<faiss::Index*>(index_ptr);
    if (index == nullptr) {
        throw std::runtime_error("index_ptr is null");
    }

    auto q_buf = q_emb.request();
    int nq, d;
    if (q_buf.ndim == 2) {
        nq = static_cast<int>(q_buf.shape[0]);
        d = static_cast<int>(q_buf.shape[1]);
    } else if (q_buf.ndim == 1) {
        nq = 1;
        d = static_cast<int>(q_buf.shape[0]);
    } else {
        throw std::runtime_error("q_emb must be 1-D (d,) or 2-D (nq, d)");
    }
    if (d != index->d) {
        throw std::runtime_error("query dimension does not match index->d");
    }
    if (ef_search < k) ef_search = k;

    auto ep_buf = entry_points.request();
    int ep_rows = 1;
    int ep_cols = 0;
    if (ep_buf.ndim == 1) {
        ep_cols = static_cast<int>(ep_buf.shape[0]);
    } else if (ep_buf.ndim == 2) {
        ep_rows = static_cast<int>(ep_buf.shape[0]);
        ep_cols = static_cast<int>(ep_buf.shape[1]);
        if (ep_rows != nq) {
            throw std::runtime_error("entry_points with ndim=2 must have shape (nq, ep)");
        }
    } else {
        throw std::runtime_error("entry_points must be 1-D or 2-D");
    }
    if (ep_cols <= 0) {
        throw std::runtime_error("entry_points is empty");
    }

    auto off_buf = offsets.request();
    auto nei_buf = neighbors.request();
    const int64_t* off = static_cast<const int64_t*>(off_buf.ptr);
    const int64_t* nei = static_cast<const int64_t*>(nei_buf.ptr);
    const int64_t* ep = static_cast<const int64_t*>(ep_buf.ptr);
    const float* q_ptr = static_cast<const float*>(q_buf.ptr);

    auto scores_out = py::array_t<float>({nq, k});
    auto indices_out = py::array_t<int64_t>({nq, k});
    auto visited_out = py::array_t<int64_t>({nq});

    float* out_scores = static_cast<float*>(scores_out.request().ptr);
    int64_t* out_indices = static_cast<int64_t*>(indices_out.request().ptr);
    int64_t* out_visited = static_cast<int64_t*>(visited_out.request().ptr);

    std::fill(out_scores, out_scores + static_cast<size_t>(nq) * k, -std::numeric_limits<float>::infinity());
    std::fill(out_indices, out_indices + static_cast<size_t>(nq) * k, static_cast<int64_t>(-1));

    const bool use_ip = index->metric_type == faiss::METRIC_INNER_PRODUCT;
    std::vector<float> vec(static_cast<size_t>(d));

    for (int qi = 0; qi < nq; ++qi) {
        const float* query = (q_buf.ndim == 2) ? (q_ptr + static_cast<size_t>(qi) * d) : q_ptr;

        std::priority_queue<std::pair<float, idx_t>, std::vector<std::pair<float, idx_t>>, CandidateMaxCmp> candidates;
        std::priority_queue<std::pair<float, idx_t>, std::vector<std::pair<float, idx_t>>, ResultMinCmp> results;
        std::unordered_set<idx_t> visited;
        visited.reserve(static_cast<size_t>(ef_search) * 4 + ep_cols * 4 + 128);

        auto add_node = [&](idx_t node) {
            if (node < 0 || node >= index->ntotal) return;
            if (visited.find(node) != visited.end()) return;
            visited.insert(node);

            index->reconstruct(node, vec.data());
            float score = use_ip ? dot_product(query, vec.data(), d) : neg_l2_distance(query, vec.data(), d);

            candidates.emplace(score, node);
            if (static_cast<int>(results.size()) < ef_search) {
                results.emplace(score, node);
            } else if (score > results.top().first) {
                results.pop();
                results.emplace(score, node);
            }
        };

        // Add initial entry points. If entry_points is 2-D, each query has its own row.
        const int64_t* ep_row = (ep_buf.ndim == 2) ? (ep + static_cast<size_t>(qi) * ep_cols) : ep;
        for (int j = 0; j < ep_cols; ++j) {
            add_node(static_cast<idx_t>(ep_row[j]));
        }

        while (!candidates.empty()) {
            auto cur = candidates.top();
            candidates.pop();
            float current_score = cur.first;
            idx_t current = cur.second;

            if (static_cast<int>(results.size()) >= ef_search && current_score < results.top().first) {
                break;
            }

            int64_t start = off[current];
            for (int j = 0; j < degree0; ++j) {
                idx_t nb = static_cast<idx_t>(nei[start + j]);
                if (nb >= 0) add_node(nb);
            }
        }

        // Move result heap to vector and sort descending.
        std::vector<std::pair<float, idx_t>> top;
        top.reserve(results.size());
        while (!results.empty()) {
            top.push_back(results.top());
            results.pop();
        }
        std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

        int limit = std::min(k, static_cast<int>(top.size()));
        for (int r = 0; r < limit; ++r) {
            out_scores[static_cast<size_t>(qi) * k + r] = top[r].first;
            out_indices[static_cast<size_t>(qi) * k + r] = static_cast<int64_t>(top[r].second);
        }
        out_visited[qi] = static_cast<int64_t>(visited.size());
    }

    return py::make_tuple(scores_out, indices_out, visited_out);
}

PYBIND11_MODULE(toploc_hnsw_search, m) {
    m.doc() = "TopLoc-HNSW custom level-0 beam search from cached entry points";
    m.def(
        "toploc_hnsw_level0_search_ptr",
        &toploc_hnsw_level0_search_ptr,
        py::arg("index_ptr"),
        py::arg("q_emb"),
        py::arg("entry_points"),
        py::arg("offsets"),
        py::arg("neighbors"),
        py::arg("degree0"),
        py::arg("k"),
        py::arg("ef_search"),
        "Batched TopLoc-HNSW level-0 search from custom entry points"
    );
}
