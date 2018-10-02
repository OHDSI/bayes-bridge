from .abstract_model import AbstractModel
import numpy as np
import numpy.random
from bayesbridge.util.simple_warnings import warn_message_only

class LogisticModel(AbstractModel):

    def __init__(self, y, X, n_trial=None):

        if n_trial is None:
            n_trial = np.ones(len(y))
            warn_message_only(
                "The numbers of trials were not specified. The binary "
                "outcome is assumed."
            )

        self.n_trial = n_trial
        self.y = y
        self.X = X
        self.name = 'logit'

    def compute_loglik_and_gradient(self, beta):
        logit_prob = self.X.dot(beta)
        predicted_prob = LogisticModel.convert_to_probability_scale(logit_prob)
        loglik = np.sum(
            self.y * logit_prob \
            - self.n_trial * np.log(1 + np.exp(logit_prob))
        )
        grad = self.X.Tdot(self.y - self.n_trial * predicted_prob)
        return loglik, grad

    def compute_hessian(self, beta):
        predicted_prob = LogisticModel.compute_predicted_prob(self.X, beta)
        weight = predicted_prob * (1 - predicted_prob)
        return - self.X.compute_fisher_info(weight)

    def get_hessian_matvec_operator(self, beta):
        predicted_prob = LogisticModel.compute_predicted_prob(self.X, beta)
        weight = predicted_prob * (1 - predicted_prob)
        hessian_op = lambda v: \
            self.X.Tdot(weight * self.X.dot(v))
        return hessian_op

    @staticmethod
    def compute_predicted_prob(X, beta, truncate=False):
        logit_prob = X.dot(beta)
        return LogisticModel.convert_to_probability_scale(logit_prob, truncate)

    @staticmethod
    def convert_to_probability_scale(logit_prob, truncate=False):
        # The flag 'truncate == True' guarantees 0 < prob < 1.
        if truncate:
            upper_bd = 36.7  # approximately - log(2 ** -53)
            lower_bd = - 709  # approximately - log(2 ** 1023)
            logit_prob[logit_prob > upper_bd] = upper_bd
            logit_prob[logit_prob < lower_bd] = lower_bd
        prob = 1 / (1 + np.exp(-logit_prob))
        return prob

    @staticmethod
    def simulate_outcome(n_trial, X, beta, seed=None):
        prob = LogisticModel.compute_predicted_prob(X, beta)
        np.random.seed(seed)
        y = np.random.binomial(n_trial, prob)
        return y