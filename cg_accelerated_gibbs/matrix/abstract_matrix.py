import numpy as np
import scipy.sparse as sparse
import abc

class AbstractMatrix():

    @abc.abstractmethod
    def dot(self, v):
        pass

    @abc.abstractmethod
    def Tdot(self, v):
        """ Multiply by the transpose of the matrix. """
        pass

    @abc.abstractmethod
    def matmul_by_diag(self, v, from_, order=None):
        """
        Computes dot(diag(v), matrix) if direc == 'left' or dot(matrix, diag(v))
        if direc == 'right'. Return a numpy or scipy.sparse array.

        Params:
        ------
        v : vector
        from_: {'left', 'right'}
        """

    @abc.abstractmethod
    def transpose(self):
        pass

    @abc.abstractmethod
    def matdot(self, another):
        """ Multiply two Matrix objects and returns a numpy array. """

    @abc.abstractmethod
    def sqnorm(self, axis=0):
        pass

    @abc.abstractmethod
    def elemwise_power(self, exponent, order=None):
        pass

    @abc.abstractmethod
    def sum(self):
        pass

    @abc.abstractmethod
    def toarray(self):
        """ Returns a 2-dimensional numpy array. """
        pass
