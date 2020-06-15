from .abstract_model import AbstractModel
import numpy as np
import scipy as sp
from warnings import warn


class CoxModel(AbstractModel):

    def __init__(self, event_time, censoring_time, design):
        """

        Parameters
        ----------
        event_time : numpy array
            The lowest values indicate the earliest events. float('inf')
            indicates right-censoring.
        censoring_time : numpy array
            float('inf') indicates uncensored observations.
        """

        if np.any(event_time[:-1] > event_time[1:]):
            raise ValueError(
                "The observations need to be sorted so that the event times are "
                "in the increasing order, from the earliest to last events."
            )

        if np.any(censoring_time[:-1] < censoring_time[1:]):
            raise ValueError(
                "The observations need to be sorted so that the censoring times "
                "are in the decreasing order, from uncensored, last censored, "
                "to the earliest censored."
            )

        n_event = len(event_time) - np.sum(np.isinf(event_time))
        risk_set_start_index, risk_set_end_index = \
            self._find_risk_set_index(
                event_time[:n_event],
                np.flip(censoring_time[n_event:])
            )
        n_appearance = CoxModel.count_risk_set_appearance(
            len(event_time), risk_set_start_index, risk_set_end_index
        )
        if not np.all(n_appearance >= 1):
            raise ValueError(
                "Some individuals never appear in the risk set. They have to be"
                "removed before using the CoxModel class.")

        self.n_event = n_event
        self.event_time = event_time
        self.censoring_time = censoring_time
        self.n_appearance_in_risk_set = n_appearance
        self.risk_set_start_index = risk_set_start_index
        self.risk_set_end_index = risk_set_end_index
        self.design = design
        self.name = 'cox'

    @staticmethod
    def preprocess_data(event_time, censoring_time, X):
        event_time, censoring_time, X = \
            CoxModel._permute_observations_by_event_and_censoring_time(
                event_time, censoring_time, X
            )
        event_time, censoring_time, X = \
            CoxModel._drop_uninformative_observations(
                event_time, censoring_time, X
            )
        return event_time, censoring_time, X

    @staticmethod
    def _permute_observations_by_event_and_censoring_time(
            event_time, censoring_time, X):
        """
        Permute the observations so that they are ordered from the earliest
        to the last to experience events, followed by the last censored to the
        earliest censored.

        Params
        ------
        event_time : numpy array
            The lowest values indicate the earliest events. float('inf')
            indicates right-censoring.
        X : numpy array or scipy sparse matrix
        """
        if not np.all(np.equal(
                event_time == float("inf"),
                censoring_time < float('inf')
            )):
            raise ValueError(
                "Either event or censoring time must be infinity for each "
                "observation."
            )

        is_sorted = (
            np.all(event_time[:-1] <= event_time[1:])
            and np.all(censoring_time[:-1] >= censoring_time[1:])
        )
        if is_sorted:
            return event_time, censoring_time, X

        warn(
            "The observations and design matrix will be sorted so that the event "
            "times are in the ascending order and censoring times in the descending order."
        )

        n_event = np.sum(event_time < float('inf'))
        event_rank = CoxModel.np_rank_by_value(event_time)
        censoring_rank = CoxModel.np_rank_by_value(censoring_time)
        sort_ind = np.concatenate((
            np.argsort(event_rank)[:n_event],
            np.argsort(- censoring_rank)[n_event:]
        ))
        assert len(np.unique(sort_ind)) == len(sort_ind)

        event_time = event_time[sort_ind]
        censoring_time = censoring_time[sort_ind]
        if sp.sparse.issparse(X):
            X = X.tocsr()[sort_ind, :]
        else:
            X = X[sort_ind, :]

        return event_time, censoring_time, X

    @staticmethod
    def _drop_uninformative_observations(event_time, censoring_time, X):

        finite_event_time = event_time[event_time < float('inf')]
        finite_censoring_time = censoring_time[censoring_time < float('inf')]

        # Exclude those censored before the first event.
        is_uninformative = (censoring_time < np.min(event_time))

        if np.any(is_uninformative):
            warn(
                "Some observations do not contribute to the likelihood, so "
                "they are being removed."
            )
            is_informative = np.logical_not(is_uninformative)
            event_time = event_time[is_informative]
            censoring_time = censoring_time[is_informative]
            X = X[is_informative, :]

        return event_time, censoring_time, X

    @staticmethod
    def np_rank_by_value(arr):
        sort_arguments = np.argsort(arr)
        rank = np.arange(len(arr))[np.argsort(sort_arguments)]
        return rank.astype('float')

    @staticmethod
    def count_risk_set_appearance(n_obs, start_index, end_index):
        """ This function assumes that the observations are already sorted in
        the way required by the class. """

        # The calculation can be done more efficiently.
        n_appearance = np.zeros(n_obs, dtype=np.int)
        for i in range(len(start_index)):
            if start_index[i] <= end_index[i]:
                n_appearance[start_index[i]:(end_index[i] + 1)] += 1
        return n_appearance

    def _find_risk_set_index(self, event_time, censoring_time):
        """ The parameters are assumed to have 'inf' removed and in the ascending order. """

        n_event = len(event_time)
        start_index = np.zeros(n_event, dtype=np.int)
        for i in range(1, n_event):
            if event_time[i - 1] == event_time[i]:
                start_index[i] = start_index[i - 1]
            else:
                start_index[i] = i

        n_censored = np.array([
            np.searchsorted(censoring_time, t) for t in event_time
        ], dtype=np.int) # Tied censoring time is considered to be in the risk set.
        end_index = len(event_time) + len(censoring_time) - 1 - n_censored

        return start_index, end_index

    def compute_loglik_and_gradient(self, beta, loglik_only=False):

        grad = None # defalt return value

        log_rel_hazard, rel_hazard, hazard_sum_over_risk_set \
            = self._compute_relative_hazard(beta)
        if np.any(hazard_sum_over_risk_set == 0.):
            loglik = - float('inf')
            return loglik, grad

        loglik = np.sum(
            log_rel_hazard[:self.n_event] - np.log(hazard_sum_over_risk_set)
        )

        if not loglik_only:
            hazard_matrix = self._HazardMultinomialProbMatrix(
                rel_hazard, hazard_sum_over_risk_set,
                self.risk_set_start_index, self.risk_set_end_index, self.n_appearance_in_risk_set
            )
            v = np.zeros(self.design.shape[0])
            v[:self.n_event] = 1
            v -= hazard_matrix.sum_over_events()
            grad = self.design.Tdot(v)

        return loglik, grad

    def _compute_relative_hazard(self, beta):

        log_rel_hazard = self.design.dot(beta)
        log_rel_hazard = CoxModel._shift_log_hazard(log_rel_hazard)

        rel_hazard = np.exp(log_rel_hazard)

        hazard_sum_over_risk_set = self._sum_over_start_end(
            rel_hazard, self.risk_set_start_index, self.risk_set_end_index
        )

        return log_rel_hazard, rel_hazard, hazard_sum_over_risk_set

    @staticmethod
    def _sum_over_start_end(arr, start_index, end_index):
        """
        Returns
        -------
        numpy array whose k-th element equals
            np.sum(arr[start_index[k]:(1 + end_index[k])])
        """
        sum_from_right = \
            np.cumsum(arr[start_index[-1]:])[end_index - start_index[-1]]
        sum_from_left = np.concatenate((
            CoxModel.np_reverse_cumsum(arr[:start_index[-1]]), [0]
        ))
        total_sum = sum_from_right + sum_from_left
        return total_sum

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

        _, rel_hazard, hazard_sum_over_risk_set \
            = self._compute_relative_hazard(beta)
        if np.any(hazard_sum_over_risk_set == 0.):
            raise ValueError(
                'Hessian operator cannot be computed likely due to an '
                'unreasonable value of regression coefficients. This could '
                'be caused by the likelihood and prior both being too weak '
                'or by a poor initialization of the Markov chain.'
            )
        W = self._HazardMultinomialProbMatrix(
            rel_hazard, hazard_sum_over_risk_set,
            self.risk_set_start_index, self.risk_set_end_index, self.n_appearance_in_risk_set
        )
        def hessian_op(beta):
            X_beta = self.design.dot(beta)
            result_vec = - self.design.Tdot(
                W.sum_over_events() * X_beta - W.Tdot(W.dot(X_beta))
            )
            return result_vec

        return hessian_op

    @staticmethod
    def simulate_outcome(X, beta, censoring_frac=.9, seed=None):
        """
        Simulate an outcome from a constant baseline hazard model i.e. the
        survival time is exponential.
        """
        if seed is not None:
            np.random.seed(seed)

        log_hazard_rate = X.dot(beta)
        log_hazard_rate = CoxModel._shift_log_hazard(log_hazard_rate)
        hazard_rate = np.exp(log_hazard_rate)
        event_time = np.random.exponential(scale=hazard_rate ** -1)

        scale = CoxModel._solve_for_exp_scale(
            np.quantile(event_time, 1 - censoring_frac), 1 - censoring_frac
        )
        censoring_time = np.random.exponential(
            scale=scale * np.ones(len(hazard_rate))
        )
        censoring_time[event_time < censoring_time] = float("inf")
        event_time[event_time >= censoring_time] = float('inf')

        return event_time, censoring_time

    @staticmethod
    def _solve_for_exp_scale(t, prob):
        """
        Computes the scale of an exponential random variable Z such that
            P(Z < t) == prob
        """
        return - t / np.log(1 - prob)

    class _HazardMultinomialProbMatrix():
        """
        Defines operations by a matrix whose each row represents the conditional
        probabilities of the event happening to the individuals in the risk set.
        """

        def __init__(self, rel_hazard, hazard_sum_over_risk_set,
                     risk_set_start_index, risk_set_end_index, n_appearance_in_risk_set):
            self.rel_hazard = rel_hazard
            self.hazard_sum_over_risk_set = hazard_sum_over_risk_set
            self.risk_set_start_index = risk_set_start_index
            self.risk_set_end_index = risk_set_end_index
            self.n_appearance_in_risk_set = n_appearance_in_risk_set
            self.n_event = len(hazard_sum_over_risk_set)


        def sum_over_events(self):
            """
            Returns the same value as the row sum of the explicitly computed
            the matrix (e.g. via the 'compute_matrix' method) but do it more
            efficiently.
            """
            normalizer_cumsum = np.cumsum(self.hazard_sum_over_risk_set ** -1)
            row_sum = normalizer_cumsum[self.n_appearance_in_risk_set - 1] \
                      * self.rel_hazard
            return row_sum

        def dot(self, v):
            return self.hazard_sum_over_risk_set ** - 1 * CoxModel._sum_over_start_end(
                self.rel_hazard * v, self.risk_set_start_index, self.risk_set_end_index
            )

        def Tdot(self, v):
            partial_inner_prod = np.cumsum(self.hazard_sum_over_risk_set ** -1 * v)
            return self.rel_hazard \
                   * partial_inner_prod[self.n_appearance_in_risk_set - 1]

        def compute_matrix(self):
            multinomial_prob = np.outer(
                self.hazard_sum_over_risk_set ** -1,
                self.rel_hazard
            )
            multinomial_prob = np.triu(multinomial_prob)
            for i in range(1, multinomial_prob.shape[0] + 1):
                if (self.risk_set_end_index[-i] + 1) >= len(self.rel_hazard):
                    break
                multinomial_prob[-i, (self.risk_set_end_index[-i] + 1):] = 0

            return multinomial_prob