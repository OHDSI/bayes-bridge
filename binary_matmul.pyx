import numpy as np
cimport numpy as np
import cython
cimport cython

ctypedef np.int32_t INT_t
ctypedef np.float_t FLOAT_t
FLOAT = np.float64

#cpdef binary_matmul(np.ndarray[INT_t, ndim=1] indices, np.ndarray[INT_t, ndim=1] indptr, np.ndarray[FLOAT_t, ndim=1] v):

@cython.boundscheck(False)
@cython.wraparound(False)
cdef binary_matmul(np.ndarray[INT_t, ndim=1] indices, np.ndarray[INT_t, ndim=1] indptr, np.ndarray[FLOAT_t, ndim=1] v):
    cdef int i, k
    cdef int m = indptr.shape[0] - 1
    cdef FLOAT_t val
    cdef np.ndarray[FLOAT_t, ndim=1] Xv = np.zeros(m, dtype=FLOAT)
    for i in range(m, nogil=True):
      val = 0
      for k in range(indptr[i], indptr[i + 1]):
        val += v[indices[k]]
      Xv[i] = val
    return Xv
