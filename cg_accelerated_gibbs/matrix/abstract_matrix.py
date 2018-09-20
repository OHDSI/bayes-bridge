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
    def compute_fisher_info(self, weight, diag_only):
        """ Computes X' diag(weight) X and returns it as a numpy array. """
        pass

    @abc.abstractmethod
    def toarray(self):
        """ Returns a 2-dimensional numpy array. """
        pass
