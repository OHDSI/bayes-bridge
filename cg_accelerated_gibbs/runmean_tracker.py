import numpy as np

class RunmeanTracker():

    def __init__(self, beta_scaled, sd_prior_samplesize=5):
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
        self.sd_prior_guess = np.ones(len(beta_scaled))
        self.n_averaged = 0
        self.runmean = {
            'beta_scaled': np.zeros(len(beta_scaled)),
            'beta_scaled_sq': np.ones(len(beta_scaled))
        }

    def update_runmean(self, beta_scaled):

        weight = 1 / (1 + self.n_averaged)
        self.runmean['beta_scaled'] = (
            weight * beta_scaled + (1 - weight) * self.runmean['beta_scaled']
        )
        self.runmean['beta_scaled_sq'] = (
            weight * beta_scaled ** 2
            + (1 - weight) * self.runmean['beta_scaled_sq']
        )
        self.n_averaged += 1

    def estimate_post_sd(self):

        beta_scaled_mean = self.runmean['beta_scaled']
        beta_scaled_sq_mean = self.runmean['beta_scaled_sq']

        if self.n_averaged > 1:
            var_estimator = self.n_averaged / (self.n_averaged - 1) * (
                beta_scaled_sq_mean - beta_scaled_mean ** 2
            )
            estimator_weight = (self.n_averaged - 1) \
                               / (
                               self.n_averaged - 1 + self.sd_prior_samplesize)
            beta_scaled_sd = np.sqrt(
                estimator_weight * var_estimator \
                + (1 - estimator_weight) * self.sd_prior_guess ** 2
            )
        else:
            beta_scaled_sd = self.sd_prior_guess

        return beta_scaled_sd