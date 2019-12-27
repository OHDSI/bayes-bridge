from .abstract_model import AbstractModel
import numpy as np
import numpy.random
from bayesbridge.util.simple_warnings import warn_message_only

class LogisticModel(AbstractModel):

    # TODO: Python crushes during the Gibbs if n_success has the second
    # dimension (instead of being a vector). Add checks for the inputs.
    def __init__(self, n_success, X, n_trial=None):

        self.check_input_validity(n_success, n_trial, X)
        if n_trial is None:
            n_trial = np.ones(len(n_success))
            warn_message_only(
                "The numbers of trials were not specified. The binary "
                "outcome is assumed."
            )

        self.n_trial = n_trial.astype('float64')
        self.n_success = n_success.astype('float64')
        self.X = X
        self.name = 'logit'

    def check_input_validity(self, n_success, n_trial, X):

        if n_trial is None:
            if np.max(n_success) > 1:
                raise ValueError(
                    "If not binary, the number of trials must be specified.")
            if not len(n_success) == X.shape[0]:
                raise ValueError(
                    "Incompatible sizes of the outcome and design matrix."
                )
            return # No need to check the rest for the default initialization.

        if not len(n_trial) == len(n_success) == X.shape[0]:
            raise ValueError(
                "Incompatible sizes of the outcome vectors and design matrix."
            )

        if np.any(n_trial <= 0):
            raise ValueError("Number of trials must be strictly positive.")

        if np.any(n_success > n_trial):
            raise ValueError(
                "Number of successes cannot be larger than that of trials.")

    def compute_loglik_and_gradient(self, beta, loglik_only=False):
        logit_prob = self.X.dot(beta)
        predicted_prob = LogisticModel.convert_to_probability_scale(logit_prob)
        loglik = np.sum(
            self.n_success * logit_prob \
            - self.n_trial * np.log(1 + np.exp(logit_prob))
        )
        if loglik_only:
            grad = None
        else:
            grad = self.X.Tdot(self.n_success - self.n_trial * predicted_prob)
        return loglik, grad

    def compute_hessian(self, beta):
        predicted_prob = LogisticModel.compute_predicted_prob(self.X, beta)
        weight = predicted_prob * (1 - predicted_prob)
        return - self.X.compute_fisher_info(weight)

    def get_hessian_matvec_operator(self, beta):
        predicted_prob = LogisticModel.compute_predicted_prob(self.X, beta)
        weight = predicted_prob * (1 - predicted_prob)
        hessian_op = lambda v: \
            - self.X.Tdot(self.n_trial * weight * self.X.dot(v))
        return hessian_op

    @staticmethod
    def compute_polya_gamma_mean(shape, tilt):
        min_magnitude = 1e-5
        pg_mean = shape.copy() / 2
        is_nonzero = (np.abs(tilt) > min_magnitude)
        pg_mean[is_nonzero] \
            *= 1 / tilt[is_nonzero] \
               * (np.exp(tilt[is_nonzero]) - 1) / (np.exp(tilt[is_nonzero]) + 1)
        return pg_mean

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