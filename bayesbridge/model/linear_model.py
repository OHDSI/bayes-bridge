from .abstract_model import AbstractModel


class LinearModel(AbstractModel):

    def __init__(self, y, X):
        self.y = y
        self.X = X
        self.name = 'linear'

    def compute_loglik_and_gradient(self, beta, loglik_only=False):
        pass

    def compute_hessian(self, beta):
        pass

    def get_hessian_matvec_operator(self, beta):
        pass

    @staticmethod
    def simulate_outcome():
        pass