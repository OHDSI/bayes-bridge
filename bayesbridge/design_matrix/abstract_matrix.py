import abc
import numpy as np

class AbstractDesignMatrix():

    def __init__(self):
        self.dot_count = 0
        self.Tdot_count = 0
        self.memoized = False
        self.X_dot_v = None # For memoization
        self.v_prev = None # For memoization

    @property
    def shape(self):
        pass

    @abc.abstractmethod
    def dot(self, v):
        pass

    @abc.abstractmethod
    def Tdot(self, v):
        """ Multiply by the transpose of the matrix. """
        pass

    def memoize_dot(self, flag=True):
        self.memoized = flag
        if self.v_prev is None:
            self.v_prev = np.full(self.shape[1], float('nan'))
        if not flag:
            self.X_dot_v = None
            self.v_prev = None

    @abc.abstractmethod
    def compute_fisher_info(self, weight, diag_only):
        """ Computes X' diag(weight) X and returns it as a numpy array. """
        pass

    @property
    def n_matvec(self):
        return self.dot_count + self.Tdot_count

    def get_dot_count(self):
        return self.dot_count, self.Tdot_count

    def reset_matvec_count(self, count=0):
        if not hasattr(count, "__len__"):
            count = 2 * [count]
        self.dot_count = count[0]
        self.Tdot_count = count[1]

    @abc.abstractmethod
    def toarray(self):
        """ Returns a 2-dimensional numpy array. """
        pass
