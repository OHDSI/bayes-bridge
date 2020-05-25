import abc


class AbstractModel():

    @property
    def n_obs(self):
        return self.X.shape[0]

    @property
    def n_pred(self):
        return self.X.shape[1]

    @property
    def intercept_added(self):
        return self.X.intercept_added

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

    @abc.abstractstaticmethod
    def simulate_outcome():
        pass