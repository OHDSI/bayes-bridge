import numpy as np
from ..util.onthefly_summarizer import OntheflySummarizer

class RegressionCoeffficientPosteriorSummarizer():

    def __init__(self, beta, gshrink, lshrunk):
        self.n_unshrunk = len(beta) - len(lshrunk)
        beta_scaled = self.scale_beta(beta, gshrink, lshrunk)
        self.beta_scaled_summarizer = OntheflySummarizer(beta_scaled)
        self.prev_pc = None

    def scale_beta(self, beta, gshrunk, lshrunk):
        beta_scaled = beta.copy()
        beta_scaled[self.n_unshrunk:] /= gshrunk * lshrunk
        return beta_scaled

    def update(self, beta, gshrunk, lshrunk):
        beta_scaled = self.scale_beta(beta, gshrunk, lshrunk)
        self.beta_scaled_summarizer.update_stats(beta_scaled)

    def update_precond_hessian_pc(self, pc):
        self.prev_pc = pc

    def extrapolate_beta_condmean(self, gshrunk, lshrunk):
        beta_condmean_guess = self.beta_scaled_summarizer.stats['mean'].copy()
        beta_condmean_guess[self.n_unshrunk:] *= gshrunk * lshrunk
        return beta_condmean_guess

    def estimate_beta_precond_scale_sd(self):
        return self.beta_scaled_summarizer.estimate_post_sd()

    def estimate_precond_hessian_pc(self):
        return self.prev_pc