import numpy as np
from ..util.onthefly_summarizer import OntheflySummarizer

class CgSamplerInitializer():

    def __init__(self, beta, tau, lam):
        self.n_unshrunk = len(beta) - len(lam)
        beta_scaled = self.scale_beta(beta, tau, lam)
        self.beta_scaled_summarizer = OntheflySummarizer(beta_scaled)

    def scale_beta(self, beta, tau, lam):
        beta_scaled = beta.copy()
        beta_scaled[self.n_unshrunk:] /= tau * lam
        return beta_scaled

    def update(self, beta, tau, lam):
        beta_scaled = self.scale_beta(beta, tau, lam)
        self.beta_scaled_summarizer.update_stats(beta_scaled)

    def guess_beta_condmean(self, tau, lam):
        beta_condmean_guess = self.beta_scaled_summarizer.stats['mean'].copy()
        beta_condmean_guess[self.n_unshrunk:] *= tau * lam
        return beta_condmean_guess

    def estimate_beta_precond_scale_sd(self):
        return self.beta_scaled_summarizer.estimate_post_sd()