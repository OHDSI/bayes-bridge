import abc

class AbstractModel():

    @abc.abstractmethod
    def compute_loglik_and_gradient(self, beta):
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