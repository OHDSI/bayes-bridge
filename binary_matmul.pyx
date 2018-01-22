import numpy as np
cimport numpy as np
import cython
cimport cython
from cython.parallel cimport prange

ctypedef np.int32_t INT_t
ctypedef np.float_t FLOAT_t
FLOAT = np.float64

def binary_matmul(X_csr, v):
  return c_binary_matmul_parallel(X_csr.indices, X_csr.indptr, v)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef c_binary_matmul(np.ndarray[INT_t, ndim=1] indices, np.ndarray[INT_t, ndim=1] indptr, np.ndarray[FLOAT_t, ndim=1] v):
    cdef int i, k
    cdef int m = indptr.shape[0] - 1
    cdef FLOAT_t val
    cdef np.ndarray[FLOAT_t, ndim=1] Xv = np.zeros(m, dtype=FLOAT)
    for i in range(m):
        val = 0
        for k in range(indptr[i], indptr[i + 1]):
            val += v[indices[k]]
        Xv[i] = val
    return Xv

@cython.boundscheck(False)
@cython.wraparound(False)
cdef c_binary_matmul_parallel(np.ndarray[INT_t, ndim=1] indices, np.ndarray[INT_t, ndim=1] indptr, np.ndarray[FLOAT_t, ndim=1] v):
    cdef int i, k
    cdef int m = indptr.shape[0] - 1
    cdef FLOAT_t val
    cdef np.ndarray[FLOAT_t, ndim=1] Xv = np.zeros(m, dtype=FLOAT)
    for i in prange(m, nogil=True):
        for k in range(indptr[i], indptr[i + 1]):
            Xv[i] += v[indices[k]]
    return Xv
