#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <faiss/IndexIVF.h>
#include <faiss/impl/FaissException.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace py = pybind11;

/*
 * Batched TopLoc IVF restricted search.
 *
 * Difference from the single-query version: q_emb may now be (nq, d).
 * The H cached centroids are reconstructed ONCE and reused across all nq
 * queries; each query independently picks its own top-nprobe subset, and a
 * single search_preassigned call covers the whole batch.
 *
 * Inputs are forced to C-contiguous float32 / int64 by the binding signature,
 * so the raw pointer arithmetic below is always valid.
 */
py::tuple toploc_ivf_search_cpp(
    faiss::IndexIVF* index,                                   // your ivf_index
    py::array_t<float,   py::array::c_style | py::array::forcecast> q_emb,       // (nq, d) or (d,)
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> cached_ids,  // (H,)
    int nprobe,                                               // your NP
    int k                                                     // your k (=10)
) {
    auto q_buf     = q_emb.request();
    auto cache_buf = cached_ids.request();

    // ── Accept (nq, d) batch or (d,) single query ────────────────
    int nq, d;
    if (q_buf.ndim == 2) {
        nq = static_cast<int>(q_buf.shape[0]);
        d  = static_cast<int>(q_buf.shape[1]);
    } else if (q_buf.ndim == 1) {
        nq = 1;
        d  = static_cast<int>(q_buf.shape[0]);
    } else {
        throw std::runtime_error("q_emb must be 1-D (d,) or 2-D (nq, d)");
    }
    if (d != index->d) {
        throw std::runtime_error("query dimension does not match index->d");
    }

    float*   q_ptr   = static_cast<float*>(q_buf.ptr);
    int64_t* ids_ptr = static_cast<int64_t*>(cache_buf.ptr);
    int      H       = static_cast<int>(cache_buf.shape[0]);   // # cached centroids

    int actual_nprobe = std::min(nprobe, H);

    // ── Step 1: Reconstruct the H cached centroid vectors ONCE ────
    // Shared across every query in the batch — this is the work we save
    // relative to scoring the full centroid set per query.
    std::vector<float> centroid_vecs(static_cast<size_t>(H) * d);
    for (int i = 0; i < H; i++) {
        index->quantizer->reconstruct(
            ids_ptr[i], centroid_vecs.data() + static_cast<size_t>(i) * d);
    }

    bool use_ip = (index->metric_type == faiss::METRIC_INNER_PRODUCT);

    // ── Per-query preassigned data, laid out row-major (nq, actual_nprobe) ─
    std::vector<int64_t> sel_centroids(static_cast<size_t>(nq) * actual_nprobe);
    std::vector<float>   sel_coarse(static_cast<size_t>(nq) * actual_nprobe);

    // Scratch reused across queries to avoid per-query allocation.
    std::vector<float> coarse_scores(H);
    std::vector<int>   order(H);

    // ── Step 2+3: For each query, score cached centroids and pick top-nprobe ─
    for (int qi = 0; qi < nq; qi++) {
        const float* q = q_ptr + static_cast<size_t>(qi) * d;

        for (int i = 0; i < H; i++) {
            const float* cv = centroid_vecs.data() + static_cast<size_t>(i) * d;
            float s = 0.0f;
            if (use_ip) {
                // inner product: higher = closer
                for (int j = 0; j < d; j++) s += cv[j] * q[j];
            } else {
                // L2: store negative squared distance so larger = closer
                for (int j = 0; j < d; j++) {
                    float diff = cv[j] - q[j];
                    s -= diff * diff;
                }
            }
            coarse_scores[i] = s;
        }

        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(
            order.begin(),
            order.begin() + actual_nprobe,
            order.end(),
            [&](int a, int b) { return coarse_scores[a] > coarse_scores[b]; }
        );

        int64_t* sc_row = sel_centroids.data() + static_cast<size_t>(qi) * actual_nprobe;
        float*   co_row = sel_coarse.data()    + static_cast<size_t>(qi) * actual_nprobe;
        for (int i = 0; i < actual_nprobe; i++) {
            sc_row[i] = ids_ptr[order[i]];
            co_row[i] = coarse_scores[order[i]];
        }
    }

    // ── Step 4: Allocate (nq, k) outputs ─────────────────────────
    auto scores_out  = py::array_t<float>  ({nq, k});
    auto indices_out = py::array_t<int64_t>({nq, k});
    float*   out_scores  = static_cast<float*>(scores_out.request().ptr);
    int64_t* out_indices = static_cast<int64_t*>(indices_out.request().ptr);

    std::fill(out_scores,  out_scores  + static_cast<size_t>(nq) * k, -1e38f);
    std::fill(out_indices, out_indices + static_cast<size_t>(nq) * k,
              static_cast<int64_t>(-1));

    // ── Step 5: One restricted search over the whole batch ───────
    // Pin the probe count explicitly so FAISS reads sel_centroids / sel_coarse
    // as exactly actual_nprobe columns per query, independent of index->nprobe.
    faiss::IVFSearchParameters params;
    params.nprobe    = actual_nprobe;
    params.max_codes = 0;

    index->search_preassigned(
        nq,                   // batch size
        q_ptr,                // queries (nq, d)
        k,                    // top-k
        sel_centroids.data(), // (nq, actual_nprobe) selected lists
        sel_coarse.data(),    // (nq, actual_nprobe) coarse distances
        out_scores,           // (nq, k) output scores
        out_indices,          // (nq, k) output ids
        false,                // store_pairs
        &params
    );

    return py::make_tuple(scores_out, indices_out);
}

// ── Pointer-based entry point (SWIG-proxy callers) ────────────────
// If pybind11 cannot auto-convert the FAISS index handed over from Python,
// pass the raw C++ pointer instead (Python: int(index.this)) and cast here.
py::tuple toploc_ivf_search_ptr(
    uintptr_t index_ptr,
    py::array_t<float,   py::array::c_style | py::array::forcecast> q_emb,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> cached_ids,
    int nprobe,
    int k
) {
    return toploc_ivf_search_cpp(
        reinterpret_cast<faiss::IndexIVF*>(index_ptr),
        q_emb, cached_ids, nprobe, k
    );
}

// ── Module registration ───────────────────────────────────────────
PYBIND11_MODULE(toploc_search, m) {
    m.doc() = "Fast batched TopLoc IVF search implemented in C++";
    m.def(
        "toploc_ivf_search",
        &toploc_ivf_search_cpp,
        py::arg("index"),
        py::arg("q_emb"),
        py::arg("cached_ids"),
        py::arg("nprobe"),
        py::arg("k"),
        "Batched TopLoc IVF restricted search (q_emb: (nq, d) or (d,))"
    );
    m.def(
        "toploc_ivf_search_ptr",
        &toploc_ivf_search_ptr,
        py::arg("index_ptr"),
        py::arg("q_emb"),
        py::arg("cached_ids"),
        py::arg("nprobe"),
        py::arg("k"),
        "Batched TopLoc IVF restricted search — index passed as raw pointer"
    );
}