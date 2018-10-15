from .abstract_model import AbstractModel
import numpy as np
from bayesbridge.util.simple_warnings import warn_message_only


class CoxModel(AbstractModel):

    def __init__(self, y, X):
        """

        Parameters
        ----------
        y : numpy array
            Either event time or event order (lowest value indicating the
            earliest event). float('inf') indicates right-censoring.
        X
        """

        if np.any(y[:-1] > y[1:]):
            raise ValueError("The event times are not sorted.")

        self.n_event = len(y) - np.sum(np.isinf(y))
        self.y = y
        self.X = X
        self.name = 'cox'

    @staticmethod
    def permute_observations_by_event_time(y, X):
        """
        Params
        ------
        y : numpy array
            Either event time or event order (lowest value indicating the
            earliest event). float('inf') indicates right-censoring.
        X : numpy array or scipy sparse matrix
        """
        event_rank = CoxModel.np_rank_by_value(y)
        sort_ind = np.argsort(event_rank)
        y = y[sort_ind]
        X = X.tocsr()[sort_ind, :]
        return y, X

    @staticmethod
    def np_rank_by_value(arr):
        sort_arguments = np.argsort(arr)
        rank = np.arange(len(arr))[np.argsort(sort_arguments)]
        return rank.astype('float')

    def compute_loglik_and_gradient(self, beta, loglik_only=False):

        log_hazard_increase, hazard_increase, sum_over_risk_set \
            = self._compute_hazard_increase(beta)

        loglik = np.sum(
            log_hazard_increase[:self.n_event] - np.log(sum_over_risk_set)
        )

        grad = None
        if not loglik_only:
            hazard_matrix = \
                self._HazardMultinomialProbMatrix(hazard_increase, sum_over_risk_set)
            v = np.zeros(self.X.shape[0])
            v[:self.n_event] = 1
            v -= hazard_matrix.sum_over_events()
            grad = self.X.Tdot(v)

        return loglik, grad

    def _compute_hazard_increase(self, beta):

        log_hazard_increase = self.X.dot(beta)
        log_hazard_increase = CoxModel._shift_log_hazard(log_hazard_increase)

        hazard_increase = np.exp(log_hazard_increase)
        sum_over_risk_set = CoxModel.np_reverse_cumsum(
            np.concatenate((
                hazard_increase[:(self.n_event - 1)],
                [np.sum(hazard_increase[(self.n_event - 1):])]
            ))
        )

        return log_hazard_increase, hazard_increase, sum_over_risk_set

    @staticmethod
    def np_reverse_cumsum(arr):
        return np.cumsum(arr[::-1])[::-1]

    @staticmethod
    def _shift_log_hazard(log_hazard_rate, log_offset=0):
        """
        Shift the values so that the max value equals 'log_offset' to
        prevent numerical under / over-flow before taking exponential.
        """
        log_hazard_rate += log_offset - np.max(log_hazard_rate)
        return log_hazard_rate

    def compute_hessian(self, beta):
        raise NotImplementedError()

    def get_hessian_matvec_operator(self, beta):

        _, hazard_increase, sum_over_risk_set \
            = self._compute_hazard_increase(beta)
        hazard_matrix = \
            self._HazardMultinomialProbMatrix(hazard_increase, sum_over_risk_set)
        W = hazard_matrix.compute_matrix()

        # TODO: Optimize the matrix-vector operation by W and W.T.
        # But it does not really matter when the number of events is small.
        def hessian_op(beta):
            X_beta = self.X.dot(beta)
            result_vec = - self.X.Tdot(
                np.sum(W, 0) * X_beta - W.T.dot(W.dot(X_beta))
            )
            return result_vec

        return hessian_op

    @staticmethod
    def simulate_outcome(X, beta, censoring_frac=.9, seed=None):
        """
        Simulate an outcome from a constant baseline hazard model i.e. the
        survival time is exponential.
        """
        np.random.seed(seed)

        log_hazard_rate = X.dot(beta)
        log_hazard_rate = CoxModel._shift_log_hazard(log_hazard_rate)
        hazard_rate = np.exp(log_hazard_rate)
        event_time = np.random.exponential(scale=hazard_rate ** -1)
        event_order = CoxModel.np_rank_by_value(event_time)
        event_order[
            event_order > censoring_frac * len(event_order)
        ] = float('inf') # Right-censoring

        return event_order

    class _HazardMultinomialProbMatrix():
        """
        Defines operations by a matrix whose each row represents the conditional
        probabilities of the event happening to the individuals in the risk set.
        """

        def __init__(self, hazard_increase, sum_over_risk_set):
            self.hazard_increase = hazard_increase
            self.sum_over_risk_set = sum_over_risk_set
            self.n_event = len(sum_over_risk_set)

        def sum_over_events(self):
            """
            Returns the same value as the row sum of the explicitly computed
            the matrix (e.g. via the 'compute_matrix' method) but do it more
            efficiently.
            """
            normalizer_cumsum = np.cumsum(self.sum_over_risk_set ** -1)
            row_sum = np.concatenate((
                normalizer_cumsum[:self.n_event]
                    * self.hazard_increase[:self.n_event],
                normalizer_cumsum[-1] * self.hazard_increase[self.n_event:]
            ))
            return row_sum

        def compute_matrix(self):
            multinomial_prob = np.outer(
                self.sum_over_risk_set ** -1,
                self.hazard_increase
            )
            multinomial_prob = np.triu(multinomial_prob)
            return multinomial_prob