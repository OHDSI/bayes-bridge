cimport cython
from libc.math cimport sin as sin_c
from libc.math cimport exp as exp_c
from libc.math cimport fabs as fabs_c
from libc.math cimport INFINITY
# from math import sqrt, log, pow

def exp(double x):
	cdef double max_exponent
	max_exponent = 709  # ~ log(2 ** 1024)
	if x > max_exponent:
		val = INFINITY
	elif x < - max_exponent:
		val = 0.
	else:
		val = exp_c(x)
	return val

def sinc(double x):
	cdef double x_sq
	if fabs_c(x) < .01:
		x_sq = x * x
		val = 1. - x_sq / 6. * (1 - x_sq / 20.)
		    # Taylor approximation with an error bounded by 2e-16
	else:
		val = sin_c(x) / x
	return val

