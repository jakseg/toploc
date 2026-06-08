#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissException.h>
#include <vector>
#include <algorithm>
#include <memory>
#include <stdexcept>

namespace py = pybind11;

/*
 * Fast batched TopLoc IVF restricted search.
 *
 * The previous version scored the H cached centroids with a hand-written
 * single-threaded loop -- that was the bottleneck that made TopLoc slower
 * than the baseline. Here that loop is replaced by a temporary FAISS
 * IndexFlat: FAISS scores the centroids with BLAS, all CPU cores, and SIMD,
 * exactly the optimized path the baseline's own coarse step uses.
 *
 * Optionally accepts pre-reconstructed centroid vectors (cached_vecs) so the
 * reconstruct step is done ONCE per conversation in Python rather than on
 * every call.
 *
 * Pipeline: score cached centroids (FAISS flat) -> pick top-nprobe per query
 *           -> search_preassigned over the whole batch (FAISS). No hand loops.
 */
py::tuple toploc_ivf_search_cpp(
    faiss::IndexIVF* index,
    py::array_t<float,   py::array::c_style | py::array::forcecast> q_emb,       // (nq, d) or (d,)
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> cached_ids,  // (H,)
    int nprobe,
    int k,
    py::object cached_vecs_obj = py::none()   // optional (H, d) float32
) {
    auto q_buf     = q_emb.request();
    auto cache_buf = cached_ids.request();

    // -- Accept (nq, d) batch or (d,) single query --
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
    int      H       = static_cast<int>(cache_buf.shape[0]);
    int actual_nprobe = std::min(nprobe, H);

    // -- Get centroid vectors (reuse if Python passed them) --
    float* cvecs_ptr = nullptr;
    std::vector<float> cvecs_storage;
    if (!cached_vecs_obj.is_none()) {
        auto cv = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cached_vecs_obj);
        auto cv_buf = cv.request();
        if (cv_buf.shape[0] != H || cv_buf.shape[1] != d) {
            throw std::runtime_error("cached_vecs shape must be (H, d)");
        }
        cvecs_ptr = static_cast<float*>(cv_buf.ptr);
    } else {
        cvecs_storage.resize(static_cast<size_t>(H) * d);
        for (int i = 0; i < H; i++) {
            index->quantizer->reconstruct(
                ids_ptr[i], cvecs_storage.data() + static_cast<size_t>(i) * d);
        }
        cvecs_ptr = cvecs_storage.data();
    }

    // -- Score cached centroids with FAISS (BLAS, all cores) --
    // This replaces the old single-threaded scoring loop.
    std::unique_ptr<faiss::IndexFlat> temp_index;
    if (index->metric_type == faiss::METRIC_INNER_PRODUCT) {
        temp_index = std::make_unique<faiss::IndexFlatIP>(d);
    } else {
        temp_index = std::make_unique<faiss::IndexFlatL2>(d);
    }
    temp_index->add(H, cvecs_ptr);

    std::vector<float>   coarse_dists(static_cast<size_t>(nq) * actual_nprobe);
    std::vector<int64_t> coarse_labels(static_cast<size_t>(nq) * actual_nprobe);
    temp_index->search(nq, q_ptr, actual_nprobe,
                       coarse_dists.data(), coarse_labels.data());

    // Map local labels (0..H-1) -> global centroid IDs
    for (size_t i = 0; i < static_cast<size_t>(nq) * actual_nprobe; i++) {
        if (coarse_labels[i] >= 0) {
            coarse_labels[i] = ids_ptr[coarse_labels[i]];
        }
    }

    // -- Allocate (nq, k) outputs --
    auto scores_out  = py::array_t<float>  ({nq, k});
    auto indices_out = py::array_t<int64_t>({nq, k});
    float*   out_scores  = static_cast<float*>(scores_out.request().ptr);
    int64_t* out_indices = static_cast<int64_t*>(indices_out.request().ptr);

    std::fill(out_scores,  out_scores  + static_cast<size_t>(nq) * k, -1e38f);
    std::fill(out_indices, out_indices + static_cast<size_t>(nq) * k,
              static_cast<int64_t>(-1));

    // -- One restricted search over the whole batch --
    faiss::IVFSearchParameters params;
    params.nprobe    = actual_nprobe;
    params.max_codes = 0;

    index->search_preassigned(
        nq, q_ptr, k,
        coarse_labels.data(), coarse_dists.data(),
        out_scores, out_indices,
        false,    // store_pairs
        &params
    );

    return py::make_tuple(scores_out, indices_out);
}

// -- Pointer-based entry point (SWIG-proxy callers) --
py::tuple toploc_ivf_search_ptr(
    uintptr_t index_ptr,
    py::array_t<float,   py::array::c_style | py::array::forcecast> q_emb,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> cached_ids,
    int nprobe,
    int k,
    py::object cached_vecs_obj = py::none()
) {
    return toploc_ivf_search_cpp(
        reinterpret_cast<faiss::IndexIVF*>(index_ptr),
        q_emb, cached_ids, nprobe, k, cached_vecs_obj
    );
}

// -- Module registration --
PYBIND11_MODULE(toploc_search, m) {
    m.doc() = "Fast batched TopLoc IVF search -- FAISS-backed scoring";
    m.def(
        "toploc_ivf_search",
        &toploc_ivf_search_cpp,
        py::arg("index"),
        py::arg("q_emb"),
        py::arg("cached_ids"),
        py::arg("nprobe"),
        py::arg("k"),
        py::arg("cached_vecs") = py::none(),
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
        py::arg("cached_vecs") = py::none(),
        "Batched TopLoc IVF restricted search -- index passed as raw pointer"
    );
}