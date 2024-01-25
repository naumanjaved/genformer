#!python
#cython: language_level=3
#cython: infer_types=True
cimport numpy as np

from cython.parallel import prange

ctypedef np.int_t DTYPE_t
ctypedef np.float64_t DTYPE_t_f
ctypedef np.str DTYPE_t_s

cimport cython
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef get_per_base_cov(np.ndarray[DTYPE_t, ndim=1] start, 
                     np.ndarray[DTYPE_t, ndim=1] end, 
                     np.ndarray[DTYPE_t, ndim=1] score, 
                     np.ndarray[DTYPE_t, ndim=1] base, 
                     np.int_t length):
    
    cdef Py_ssize_t k
    
    cdef long[:] start_arr = start
    cdef long[:] end_arr = end 
    cdef long[:] score_arr = score 
    cdef long[:] base_arr = base
    
    cdef long num_bins = length
    
    for k in prange(num_bins, nogil=True):
        base_arr[start_arr[k]:end_arr[k]] = score_arr[k]

    return base_arr

cimport cython
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef get_per_base_score_f(np.ndarray[DTYPE_t, ndim=1] start, 
                     np.ndarray[DTYPE_t, ndim=1] end, 
                     np.ndarray[DTYPE_t_f, ndim=1] score, 
                     np.ndarray[DTYPE_t_f, ndim=1] base, 
                     np.int_t length):
    
    cdef Py_ssize_t k
    
    cdef long[:] start_arr = start
    cdef long[:] end_arr = end 
    cdef double[:] score_arr = score 
    cdef double[:] base_arr = base
    
    cdef long num_bins = length
    
    for k in prange(num_bins, nogil=True):
        base_arr[start_arr[k]:end_arr[k]] = score_arr[k]

    return base_arr


