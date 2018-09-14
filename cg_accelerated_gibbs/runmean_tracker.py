import numpy as np

class RunmeanTracker():

    def __init__(self, theta, sd_prior_samplesize=5):
        """

        Params
        ------
        init: dict
        sd_prior_samplesize: int
            Weight on the initial estimate of the posterior standard
            deviation; the estimate is treated as if it is an average of
            'prior_samplesize' previous values.
        """
        self.sd_prior_samplesize = sd_prior_samplesize
        self.sd_prior_guess = np.ones(len(theta))
        self.n_averaged = 0
        self.runmean = {
            'mean': np.zeros(len(theta)),
            'square': np.ones(len(theta))
        }

    def update_runmean(self, theta):

        weight = 1 / (1 + self.n_averaged)
        self.runmean['mean'] = (
            weight * theta + (1 - weight) * self.runmean['mean']
        )
        self.runmean['square'] = (
            weight * theta ** 2
            + (1 - weight) * self.runmean['square']
        )
        self.n_averaged += 1

    def estimate_post_sd(self):

        mean = self.runmean['mean']
        sec_moment = self.runmean['square']

        if self.n_averaged > 1:
            var_estimator = self.n_averaged / (self.n_averaged - 1) * (
                sec_moment - mean ** 2
            )
            estimator_weight = (self.n_averaged - 1) \
                               / (
                               self.n_averaged - 1 + self.sd_prior_samplesize)
            sd_estimator = np.sqrt(
                estimator_weight * var_estimator \
                + (1 - estimator_weight) * self.sd_prior_guess ** 2
            )
        else:
            sd_estimator = self.sd_prior_guess

        return sd_estimator