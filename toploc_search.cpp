#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <faiss/IndexIVF.h>
#include <faiss/impl/FaissException.h>
#include <vector>
#include <algorithm>
#include <numeric>

namespace py = pybind11;

/*
 * This replaces your entire toploc_ivf_search() Python function.
 * 
 * Everything that was manual numpy math in Python
 * now happens inside FAISS C++ memory directly.
 * No crossing back and forth anymore.
 */
py::tuple toploc_ivf_search_cpp(
    faiss::IndexIVF* index,       // your ivf_index
    py::array_t<float> q_emb,     // your q_emb  (shape: 1 x d)
    py::array_t<int64_t> cached_ids, // your conv_cache[conv_id]
    int nprobe,                   // your NP
    int k                         // your k (=10)
) {
    // ── Read inputs ──────────────────────────────────────────────
    auto q_buf     = q_emb.request();
    auto cache_buf = cached_ids.request();

    float*   q_ptr    = static_cast<float*>(q_buf.ptr);
    int64_t* ids_ptr  = static_cast<int64_t*>(cache_buf.ptr);
    int      H        = cache_buf.shape[0];   // how many cached centroids
    int      d        = index->d;             // vector dimension

    // ── Step 1: Fetch centroid vectors from FAISS ─────────────────
    // This replaces your get_centroid_vectors() call in Python.
    // Everything stays in C++ memory — no Python numpy array created.
    std::vector<float> centroid_vecs(H * d);
    for (int i = 0; i < H; i++) {
        index->quantizer->reconstruct(ids_ptr[i], centroid_vecs.data() + i * d);
    }

    // ── Step 2: Score each cached centroid against the query ──────
    // This replaces your manual numpy dot product / L2 math in Python:
    //   coarse = (centroid_vecs @ q_emb.T).squeeze()   ← gone
    //   coarse = ((centroid_vecs - q_emb)**2).sum()    ← gone
    bool use_ip = (index->metric_type == faiss::METRIC_INNER_PRODUCT);
    std::vector<float> coarse_scores(H);

    for (int i = 0; i < H; i++) {
        float score = 0.0f;
        float* cv = centroid_vecs.data() + i * d;
        if (use_ip) {
            // inner product: higher = closer
            for (int j = 0; j < d; j++) score += cv[j] * q_ptr[j];
        } else {
            // L2: lower = closer, store as negative so argmax works
            for (int j = 0; j < d; j++) {
                float diff = cv[j] - q_ptr[j];
                score -= diff * diff;
            }
        }
        coarse_scores[i] = score;
    }

    // ── Step 3: Pick top nprobe centroids from the cached set ─────
    // This replaces your np.argpartition + np.argsort in Python:
    //   top_local = np.argpartition(-coarse, nprobe)[:nprobe]  ← gone
    //   top_local = top_local[np.argsort(-coarse[top_local])]  ← gone
    int actual_nprobe = std::min(nprobe, H);
    std::vector<int> order(H);
    std::iota(order.begin(), order.end(), 0);   // [0, 1, 2, ..., H-1]
    std::partial_sort(
        order.begin(),
        order.begin() + actual_nprobe,
        order.end(),
        [&](int a, int b) { return coarse_scores[a] > coarse_scores[b]; }
    );

    // ── Step 4: Build the arrays search_preassigned needs ─────────
    // This replaces your reshape/astype calls in Python:
    //   sel_centroids = np.asarray(...).astype("int64").reshape(1,-1)  ← gone
    //   sel_coarse    = coarse[top_local].astype("float32").reshape(1,-1) ← gone
    std::vector<int64_t> sel_centroids(actual_nprobe);
    std::vector<float>   sel_coarse(actual_nprobe);
    for (int i = 0; i < actual_nprobe; i++) {
        sel_centroids[i] = ids_ptr[order[i]];
        sel_coarse[i]    = coarse_scores[order[i]];
    }

    // ── Step 5: Run the restricted search ────────────────────────
    // This is the same search_preassigned call you had in Python,
    // but now it never leaves C++ memory to get here.
    // No try/except needed — we call the C++ API directly.
    std::vector<float>   out_scores(k, -1e38f);
    std::vector<int64_t> out_indices(k, -1);

    index->search_preassigned(
        1,                    // nq = 1 query
        q_ptr,                // query vector
        k,                    // top-k
        sel_centroids.data(), // which centroids to search
        sel_coarse.data(),    // their coarse distances
        out_scores.data(),    // output scores
        out_indices.data(),   // output doc indices
        false                 // store_pairs = false
    );

    // ── Return as numpy arrays (same shape as your Python version) ─
    // scores  shape: (1, k)
    // indices shape: (1, k)
    auto scores_out  = py::array_t<float>  ({1, k});
    auto indices_out = py::array_t<int64_t>({1, k});

    auto s_buf = scores_out.request();
    auto i_buf = indices_out.request();

    std::copy(out_scores.begin(),  out_scores.end(),
              static_cast<float*>(s_buf.ptr));
    std::copy(out_indices.begin(), out_indices.end(),
              static_cast<int64_t*>(i_buf.ptr));

    return py::make_tuple(scores_out, indices_out);
}

// ── Pointer-based entry point ─────────────────────────────────────
// Some FAISS Python builds (e.g. the conda-forge wheel used for the
// laptop demo) hand the index over as a SWIG proxy that pybind11 cannot
// auto-convert to faiss::IndexIVF*. In that case the caller passes the
// raw C++ pointer instead (Python: int(index.this)) and we cast it here.
// The actual algorithm (toploc_ivf_search_cpp) is unchanged.
py::tuple toploc_ivf_search_ptr(
    uintptr_t index_ptr,
    py::array_t<float> q_emb,
    py::array_t<int64_t> cached_ids,
    int nprobe,
    int k
) {
    return toploc_ivf_search_cpp(
        reinterpret_cast<faiss::IndexIVF*>(index_ptr),
        q_emb, cached_ids, nprobe, k
    );
}

// ── Register the function so Python can import it ─────────────────
// This is what lets you write:
//   from toploc_search import toploc_ivf_search
// in your Python file — it looks identical to before.
PYBIND11_MODULE(toploc_search, m) {
    m.doc() = "Fast TopLoc IVF search implemented in C++";
    m.def(
        "toploc_ivf_search",      // name Python sees
        &toploc_ivf_search_cpp,   // actual C++ function
        py::arg("index"),
        py::arg("q_emb"),
        py::arg("cached_ids"),
        py::arg("nprobe"),
        py::arg("k"),
        "TopLoc IVF restricted search — C++ implementation"
    );
    m.def(
        "toploc_ivf_search_ptr",  // pointer-based variant for SWIG-proxy callers
        &toploc_ivf_search_ptr,
        py::arg("index_ptr"),
        py::arg("q_emb"),
        py::arg("cached_ids"),
        py::arg("nprobe"),
        py::arg("k"),
        "TopLoc IVF restricted search — index passed as raw pointer"
    );
}