import numpy as np
import scipy as sp
import scipy.sparse
from .cg_sampler import ConjugateGradientSampler
from .reg_coef_posterior_summarizer import RegressionCoeffficientPosteriorSummarizer
from .direct_gaussian_sampler import generate_gaussian_with_weight
from . import hamiltonian_monte_carlo as hmc


class SparseRegressionCoefficientSampler():

    def __init__(self, init, prior_sd_for_unshrunk, sampling_method):

        self.prior_sd_for_unshrunk = prior_sd_for_unshrunk
        self.n_unshrunk = len(prior_sd_for_unshrunk)

        # Object for keeping track of running average.
        self.regcoef_summarizer = RegressionCoeffficientPosteriorSummarizer(
            init['beta'], init['global_shrinkage'], init['local_shrinkage']
        )
        if sampling_method == 'cg':
            self.cg_sampler = ConjugateGradientSampler(self.n_unshrunk)

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
            self, y, X, obs_prec, gshrink, lshrink,
            method='cg', precond_blocksize=0):
        """
        Param:
        ------
            X: Matrix object
            beta_init: vector
                Used when when method == 'cg' as the starting value of the
                preconditioned conjugate gradient algorithm.
            method: {'direct', 'cg'}
                If 'direct', a sample is generated using a direct method based on the
                direct linear algebra. If 'cg', the preconditioned conjugate gradient
                sampler is used.

        """
        # TODO: Comment on the form of the posterior.

        v = X.Tdot(obs_prec * y)
        prior_sd = np.concatenate((
            self.prior_sd_for_unshrunk, gshrink * lshrink
        ))
        prior_prec_sqrt = 1 / prior_sd

        info = {}
        if method == 'direct':
            beta = generate_gaussian_with_weight(
                X, obs_prec, prior_prec_sqrt, v)

        elif method == 'cg':
            # TODO: incorporate an automatic calibration of 'maxiter'.
            beta_condmean_guess = \
                self.regcoef_summarizer.extrapolate_beta_condmean(gshrink, lshrink)
            beta_precond_scale_sd = self.regcoef_summarizer.estimate_beta_precond_scale_sd()
            beta, cg_info = self.cg_sampler.sample(
                X, obs_prec, prior_prec_sqrt, v,
                beta_init=beta_condmean_guess,
                precond_by='prior+block', precond_blocksize=precond_blocksize,
                beta_scaled_sd=beta_precond_scale_sd,
                maxiter=500, atol=10e-6 * np.sqrt(X.shape[1])
            )
            self.regcoef_summarizer.update(beta, gshrink, lshrink)
            info['n_cg_iter'] = cg_info['n_iter']

        else:
            raise NotImplementedError()

        return beta, info

    def sample_by_hmc(self, beta, gshrink, lshrink, model, max_step=500):

        precond_scale = self.compute_preconditioning_scale(gshrink, lshrink)
        precond_prior_prec = np.concatenate((
            (self.prior_sd_for_unshrunk / precond_scale[:self.n_unshrunk]) ** -2,
            np.ones(len(lshrink))
        ))

        beta_condmean_guess = \
            self.regcoef_summarizer.extrapolate_beta_condmean(gshrink, lshrink)
        max_curvature = self.compute_precond_hessian_curvature(
            beta_condmean_guess, model, precond_scale, precond_prior_prec
        )

        approx_stability_limit = 2 / np.sqrt(max_curvature)
        if model.name == 'cox':
            adjustment_factor = .3
        else:
            adjustment_factor = .5
        stepsize_upper_limit = adjustment_factor * approx_stability_limit
            # The multiplicative factors may require adjustment.
        dt = np.random.uniform(.5, 1) * stepsize_upper_limit
        n_step = np.ceil(1 / dt * np.random.uniform(.8, 1.)).astype('int')
        n_step = min(n_step, max_step)

        beta_precond = beta / precond_scale
        def f(beta_precond):
            beta = beta_precond * precond_scale
            logp, grad_wrt_beta = model.compute_loglik_and_gradient(beta)
            grad = precond_scale * grad_wrt_beta # Chain rule.
            logp += np.sum(- precond_prior_prec * beta_precond ** 2) / 2
            grad += - precond_prior_prec * beta_precond
            return logp, grad

        beta_precond, hmc_info = \
            hmc.generate_next_state(f, dt, n_step, beta_precond)
        beta = beta_precond * precond_scale

        info = {
            key: hmc_info[key]
            for key in ['accepted', 'accept_prob', 'n_grad_evals']
        }
        info['n_integrator_step'] = n_step
        info['stepsize'] = dt
        info['stability_limit_est'] = approx_stability_limit
        return beta, info

    def compute_preconditioning_scale(self, gshrink, lshrink):

        # TODO: this piece of codes for computing the preconditioned Hessian is
        # very similar to the CG sampler preconditioning. Perhaps we can unify them?

        beta_precond_post_sd = \
            self.regcoef_summarizer.estimate_beta_precond_scale_sd()

        precond_scale = np.ones(len(beta_precond_post_sd))
        precond_scale[self.n_unshrunk:] = gshrink * lshrink
        if self.n_unshrunk > 0:
            target_sd_scale = 2.
            precond_scale[:self.n_unshrunk] = \
                target_sd_scale * beta_precond_post_sd[:self.n_unshrunk]

        return precond_scale

    def compute_precond_hessian_curvature(
            self, beta_location, model, precond_scale, precond_prior_prec):

        loglik_hessian_matvec = model.get_hessian_matvec_operator(beta_location)
        precond_hessian_matvec = lambda beta: \
            precond_prior_prec * beta \
            - precond_scale * loglik_hessian_matvec(precond_scale * beta)
        precond_hessian_op = sp.sparse.linalg.LinearOperator(
            (len(beta_location), len(beta_location)), precond_hessian_matvec
        )
        eigval = sp.sparse.linalg.eigsh(
            precond_hessian_op, k=1, tol=.1, return_eigenvectors=False)
            # We don't need a high (relative) accuracy.
        max_curvature = eigval[0]
        return max_curvature