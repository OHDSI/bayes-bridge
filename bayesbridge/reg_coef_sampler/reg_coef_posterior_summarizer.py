import numpy as np
from ..util.onthefly_summarizer import OntheflySummarizer

class RegressionCoeffficientPosteriorSummarizer():

    def __init__(self, beta, gshrink, lshrunk, pc_summary_method='average'):
        self.n_unshrunk = len(beta) - len(lshrunk)
        beta_scaled = self.scale_beta(beta, gshrink, lshrunk)
        self.beta_scaled_summarizer = OntheflySummarizer(beta_scaled)
        self.pc_summarizer = DirectionSummarizer(pc_summary_method)

    def scale_beta(self, beta, gshrunk, lshrunk):
        beta_scaled = beta.copy()
        beta_scaled[self.n_unshrunk:] /= gshrunk * lshrunk
        return beta_scaled

    def update(self, beta, gshrunk, lshrunk):
        beta_scaled = self.scale_beta(beta, gshrunk, lshrunk)
        self.beta_scaled_summarizer.update_stats(beta_scaled)

    def update_precond_hessian_pc(self, pc):
        self.pc_summarizer.update(pc)

    def extrapolate_beta_condmean(self, gshrunk, lshrunk):
        beta_condmean_guess = self.beta_scaled_summarizer.stats['mean'].copy()
        beta_condmean_guess[self.n_unshrunk:] *= gshrunk * lshrunk
        return beta_condmean_guess

    def estimate_beta_precond_scale_sd(self):
        return self.beta_scaled_summarizer.estimate_post_sd()

    def estimate_precond_hessian_pc(self):
        return self.pc_summarizer.get_mean()


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