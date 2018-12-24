import abc

class AbstractDesignMatrix():

    def __init__(self, X):
        self.dot_count = 0
        self.Tdot_count = 0
        self.shape = X.shape

    @abc.abstractmethod
    def dot(self, v):
        self.dot_count += 1

    @abc.abstractmethod
    def Tdot(self, v):
        """ Multiply by the transpose of the matrix. """
        self.Tdot_count += 1

    @abc.abstractmethod
    def compute_fisher_info(self, weight, diag_only):
        """ Computes X' diag(weight) X and returns it as a numpy array. """
        pass

    @property
    def n_matvec(self):
        return self.dot_count + self.Tdot_count

    def get_dot_count(self):
        """ Returns a 2-dimensional numpy array. """
        return self.dot_count, self.Tdot_count

    def reset_matvec_count(self, count=0):
        """ Returns a 2-dimensional numpy array. """
        if not hasattr(count, "__len__"):
            count = 2 * [count]
        self.dot_count = count[0]
        self.Tdot_count = count[1]

    @abc.abstractmethod
    def toarray(self):
        """ Returns a 2-dimensional numpy array. """
        pass
