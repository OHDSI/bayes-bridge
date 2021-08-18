from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer
from numpy.random cimport bitgen_t
from numpy.random.bit_generator cimport BitGenerator, bitgen_t
from numpy.random.c_distributions cimport random_standard_uniform


cdef double random_uniform(BitGenerator bit_generator):
    """
    Generate a random value from a uniform(0,1) distribution.

    Parameters
    ----------
    bit_generator : BitGenerator
        Numpy BitGenerator object. This object is *not* locked during generation since the
        sampling runs on a single thread and performance is much better without locking/releasing.

    Returns
    -------
    double
        Random number.
    """
    cdef bitgen_t *rng
    cdef const char *capsule_name = "BitGenerator"
    capsule = bit_generator.capsule
    rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)
    return random_standard_uniform(rng)
