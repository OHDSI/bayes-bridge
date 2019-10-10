import numpy as np

class RegressionCoeffficientPosteriorSummarizer():

    def __init__(self, n_coef, n_unshrunk, regularizing_slab_size,
                 pc_summary_method='average'):
        self.n_unshrunk = n_unshrunk
        self.beta_scaled_summarizer = OntheflySummarizer(n_coef)
        self.slab_size = regularizing_slab_size
        self.pc_summarizer = DirectionSummarizer(pc_summary_method)

    def scale_beta(self, beta, gscale, lscale):
        beta_scaled = beta.copy()
        beta_scaled[self.n_unshrunk:] \
            /= self.compute_prior_scale(gscale, lscale)
        return beta_scaled

    def update(self, beta, gscale, lscale):
        beta_scaled = self.scale_beta(beta, gscale, lscale)
        self.beta_scaled_summarizer.update_stats(beta_scaled)

    def update_precond_hessian_pc(self, pc):
        self.pc_summarizer.update(pc)

    def extrapolate_beta_condmean(self, gscale, lscale):
        beta_condmean_guess = self.beta_scaled_summarizer.stats['mean'].copy()
        beta_condmean_guess[self.n_unshrunk:] \
            *= self.compute_prior_scale(gscale, lscale)
        return beta_condmean_guess

    def estimate_beta_precond_scale_sd(self):
        return self.beta_scaled_summarizer.estimate_post_sd()

    def estimate_precond_hessian_pc(self):
        return self.pc_summarizer.get_mean()

    def compute_prior_scale(self, gscale, lscale):
        """ Compute the regularized prior scale in a numerically stable way. """
        unreg_prior_scale = gscale * lscale
        return unreg_prior_scale \
               / np.sqrt(1 + (unreg_prior_scale / self.slab_size) ** 2)

class DirectionSummarizer():

    def __init__(self, summary_method):
        """
        Parameters
        ----------
        summary_method: str, {'average', 'previous'}
        """
        self.method = summary_method
        self.n_averaged = 0
        self.v = None

    def update(self, v):
        if self.n_averaged == 0 or self.method == 'previous':
            self.v = v
        else:
            v *= np.sign(np.inner(self.v, v))
            weight = 1 / (1 + self.n_averaged)
            self.v = weight * v + (1 - weight) * self.v
        self.n_averaged += 1

    def get_mean(self):
        return self.v


class OntheflySummarizer():
    """
    Carries out online updates of the mean, variance, and other statistics of a
    random sequence.
    """

    def __init__(self, n_param, sd_prior_samplesize=5):
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
        self.sd_prior_guess = np.ones(n_param)
        self.n_averaged = 0
        self.stats = {
            'mean': np.zeros(n_param),
            'square': np.ones(n_param)
        }

    def update_stats(self, theta):

        weight = 1 / (1 + self.n_averaged)
        self.stats['mean'] = (
            weight * theta + (1 - weight) * self.stats['mean']
        )
        self.stats['square'] = (
            weight * theta ** 2
            + (1 - weight) * self.stats['square']
        )
        self.n_averaged += 1

    def estimate_post_sd(self):

        # TODO: implment Welford's algorithm for better numerical accuracy.
        mean = self.stats['mean']
        sec_moment = self.stats['square']

        if self.n_averaged > 1:
            var_estimator = self.n_averaged / (self.n_averaged - 1) * (
                sec_moment - mean ** 2
            )
            estimator_weight = (self.n_averaged - 1) \
                / (self.n_averaged - 1 + self.sd_prior_samplesize)
            sd_estimator = np.sqrt(
                estimator_weight * var_estimator \
                + (1 - estimator_weight) * self.sd_prior_guess ** 2
            )
        else:
            sd_estimator = self.sd_prior_guess

        return sd_estimator