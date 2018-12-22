from .abstract_model import AbstractModel
import math
import numpy as np


class LinearModel(AbstractModel):

    def __init__(self, y, X):
        self.y = y
        self.X = X
        self.name = 'linear'

    def compute_loglik_and_gradient(self, beta, obs_prec, loglik_only=False):
        X_beta = self.X.dot(beta)
        loglik = (
            len(self.y) * math.log(obs_prec) / 2
            - obs_prec * np.sum((self.y - X_beta) ** 2) / 2
        )
        if loglik_only:
            grad = None
        else:
            grad = obs_prec * self.X.Tdot(self.y - X_beta)
        return loglik, grad

    def compute_hessian(self, beta):
        pass

    def get_hessian_matvec_operator(self, beta):
        pass

    @staticmethod
    def simulate_outcome():
        pass