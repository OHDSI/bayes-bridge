import numpy as np
from ..util.onthefly_summarizer import OntheflySummarizer

class RegressionCoeffficientPosteriorSummarizer():

    def __init__(self, beta, gshrink, lshrunk):
        self.n_unshrunk = len(beta) - len(lshrunk)
        beta_scaled = self.scale_beta(beta, gshrink, lshrunk)
        self.beta_scaled_summarizer = OntheflySummarizer(beta_scaled)

    def scale_beta(self, beta, gshrunk, lshrunk):
        beta_scaled = beta.copy()
        beta_scaled[self.n_unshrunk:] /= gshrunk * lshrunk
        return beta_scaled

    def update(self, beta, gshrunk, lshrunk):
        beta_scaled = self.scale_beta(beta, gshrunk, lshrunk)
        self.beta_scaled_summarizer.update_stats(beta_scaled)

    def guess_beta_condmean(self, gshrunk, lshrunk):
        beta_condmean_guess = self.beta_scaled_summarizer.stats['mean'].copy()
        beta_condmean_guess[self.n_unshrunk:] *= gshrunk * lshrunk
        return beta_condmean_guess

    def estimate_beta_precond_scale_sd(self):
        return self.beta_scaled_summarizer.estimate_post_sd()