import numpy as np
from ..sparse_dense_matrix_operators \
    import elemwise_power, left_matmul_by_diag, right_matmul_by_diag, \
    choose_optimal_format_for_matvec
from .cg_sampler import ConjugateGradientSampler
from .cg_sampler_initializer import CgSamplerInitializer
from .direct_gaussian_sampler import generate_gaussian_with_weight

class SparseRegressionCoefficientSampler():

    def __init__(self, init, prior_sd_for_unshrunk, mvnorm_method):

        self.prior_sd_for_unshrunk = prior_sd_for_unshrunk
        self.n_unshrunk = len(prior_sd_for_unshrunk)

        # Object for keeping track of running average.
        if mvnorm_method == 'pcg':
            self.cg_sampler = ConjugateGradientSampler(self.n_unshrunk)
            self.cg_initalizer = CgSamplerInitializer(
                init['beta'], init['global_shrinkage'], init['local_shrinkage']
            )

    def get_internal_state(self):
        state = {}
        attr = 'cg_initializer'
        if hasattr(self, attr):
            state[attr] = getattr(self, attr)
        return state

    def set_internal_state(self, state):
        attr = 'cg_initializer'
        if hasattr(self, attr):
            setattr(self, attr, state[attr])

    def sample_gaussian_posterior(
            self, y, X_row_major, X_col_major, obs_prec, gshrink, lshrink,
            method='pcg', precond_blocksize=0):
        """
        Param:
        ------
            X_col_major: None if X is dense, sparse csc matrix otherwise
            beta_init: vector
                Used when when method == 'pcg' as the starting value of the
                preconditioned conjugate gradient algorithm.
            method: {'dense', 'pcg'}
                If 'dense', a sample is generated using a direct method based on the
                dense linear algebra. If 'pcg', the preconditioned conjugate gradient
                sampler is used.

        """
        # TODO: Comment on the form of the posterior.

        _, X_T = choose_optimal_format_for_matvec(X_row_major, X_col_major)
        v = X_T.dot(obs_prec * y)
        prior_sd = np.concatenate((
            self.prior_sd_for_unshrunk, gshrink * lshrink
        ))
        prior_prec_sqrt = 1 / prior_sd

        if method == 'dense':
            beta = generate_gaussian_with_weight(
                X_row_major, obs_prec, prior_prec_sqrt, v)
            n_pcg_iter = np.nan

        elif method == 'pcg':
            # TODO: incorporate an automatic calibration of 'maxiter' and 'atol' to
            # control the error in the MCMC output.
            beta_condmean_guess = \
                self.cg_initalizer.guess_beta_condmean(gshrink, lshrink)
            beta_precond_scale_sd = self.cg_initalizer.estimate_beta_precond_scale_sd()
            beta, cg_info = self.cg_sampler.sample(
                X_row_major, X_col_major, obs_prec, prior_prec_sqrt, v,
                beta_init=beta_condmean_guess,
                precond_by='prior+block', precond_blocksize=precond_blocksize,
                beta_scaled_sd=beta_precond_scale_sd,
                maxiter=500, atol=10e-4
            )
            self.cg_initalizer.update(beta, gshrink, lshrink)
            n_pcg_iter = cg_info['n_iter']

        else:
            raise NotImplementedError()

        return beta, n_pcg_iter