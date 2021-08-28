import abc


class AbstractModel():

    @property
    def n_obs(self):
        return self.design.shape[0]

    @property
    def n_pred(self):
        return self.design.shape[1]

    @property
    def intercept_added(self):
        return self.design.intercept_added

    @abc.abstractmethod
    def compute_loglik_and_gradient(self, beta, loglik_only=False):
        pass

    @abc.abstractmethod
    def compute_hessian(self, beta):
        pass

    @abc.abstractmethod
    def get_hessian_matvec_operator(self, beta):
        pass

    @abc.abstractmethod
    def get_hessian_matvec_operator(self, beta):
        pass

    @abc.abstractmethod
    def calc_intercept_mle(self):
        """ Calculate MLE for intercept assuming other coefficients are zero. """
        pass

    @staticmethod
    @abc.abstractmethod
    def simulate_outcome():
        pass