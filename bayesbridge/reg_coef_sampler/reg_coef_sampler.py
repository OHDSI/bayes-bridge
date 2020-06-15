import math
import numpy as np
import scipy as sp
import scipy.sparse
from warnings import warn
from functools import partial
from .cg_sampler import ConjugateGradientSampler
from .reg_coef_posterior_summarizer import RegressionCoeffficientPosteriorSummarizer
from .direct_gaussian_sampler import generate_gaussian_with_weight
from .hamiltonian_monte_carlo import hmc
from .hamiltonian_monte_carlo.nuts import NoUTurnSampler
from .hamiltonian_monte_carlo.stepsize_adapter \
    import HamiltonianBasedStepsizeAdapter


class SparseRegressionCoefficientSampler():

    def __init__(self, n_coef, prior_sd_for_unshrunk, sampling_method,
                 stability_estimate_stabilized=False,
                 regularizing_slab_size=float('inf')):

        self.prior_sd_for_unshrunk = prior_sd_for_unshrunk
        self.n_unshrunk = len(prior_sd_for_unshrunk)
        self.regularizing_slab_size = regularizing_slab_size

        # Object for keeping track of running average.
        self.regcoef_summarizer = RegressionCoeffficientPosteriorSummarizer(
            n_coef, self.n_unshrunk, regularizing_slab_size,
            pc_summary_method='average'
        )
        if sampling_method == 'cg':
            self.cg_sampler = ConjugateGradientSampler(self.n_unshrunk)
        elif sampling_method in ['hmc', 'nuts']:
            self.stability_adjustment_adapter = \
                HamiltonianBasedStepsizeAdapter(init_stepsize=.3, target_accept_prob=.95)
            self.stability_est_stabilizer = StabilityEstimateStabilizer()
            self.stability_est_stabilized = stability_estimate_stabilized
        self._sampling_info_attributes = [
            'regcoef_summarizer',
            'stability_adjustment_adapter',
            'stability_estimate_stabilizer'
        ] # Names of the attributes tracking info from previous sampling iterations.

    def get_internal_state(self):
        state = {}
        for attr in self._sampling_info_attributes:
            if hasattr(self, attr):
                state[attr] = getattr(self, attr)
        return state

    def set_internal_state(self, state):
        for attr in self._sampling_info_attributes:
            if hasattr(self, attr):
                setattr(self, attr, state[attr])

    def sample_gaussian_posterior(
            self, y, design, obs_prec, gscale, lscale, method='cg'):
        """
        Parameters
        ----------
        beta_init: vector
            Used when when method == 'cg' as the starting value of the
            preconditioned conjugate gradient algorithm.
        method: {'cholesky', 'cg'}
            If 'cholesky', a sample is generated using a cholesky method based on the
            cholesky linear algebra. If 'cg', the preconditioned conjugate gradient
            sampler is used.
        """
        # TODO: Comment on the form of the posterior.

        v = design.Tdot(obs_prec * y)
        prior_shrunk_scale = self.compute_prior_shrunk_scale(gscale, lscale)
        prior_sd = np.concatenate((
            self.prior_sd_for_unshrunk, prior_shrunk_scale
        ))
        prior_prec_sqrt = 1 / prior_sd

        info = {}
        if method == 'cholesky':
            beta = generate_gaussian_with_weight(
                design, obs_prec, prior_prec_sqrt, v)

        elif method == 'cg':
            beta_condmean_guess = \
                self.regcoef_summarizer.extrapolate_beta_condmean(gscale, lscale)
            beta_precond_scale_sd = self.regcoef_summarizer.estimate_beta_precond_scale_sd()
            beta, cg_info = self.cg_sampler.sample(
                design, obs_prec, prior_prec_sqrt, v,
                beta_init=beta_condmean_guess,
                precond_by='prior',
                beta_scaled_sd=beta_precond_scale_sd,
                maxiter=500, atol=10e-6 * np.sqrt(design.shape[1])
            )
            self.regcoef_summarizer.update(beta, gscale, lscale)
            info['n_cg_iter'] = cg_info['n_iter']

        else:
            raise NotImplementedError()

        return beta, info

    def sample_by_hmc(
            self, beta, gscale, lscale, model, method='hmc', max_step=512):
        # TODO: allow for a fixed stepsize (w/o adaptation)?

        beta_precond_post_sd = \
            self.regcoef_summarizer.estimate_beta_precond_scale_sd()
        precond_scale, precond_prior_prec = \
            self.compute_preconditioning_scale(
                gscale, lscale, beta_precond_post_sd, self.prior_sd_for_unshrunk
            )

        # Calibrate the stepsize by estimating the stability limit.
        approx_stability_limit, n_hessian_matvec \
            = self.compute_stability_limit(
                gscale, lscale, model, precond_scale, precond_prior_prec
            )
        if self.stability_est_stabilized:
            pre_stabilization = approx_stability_limit
            approx_stability_limit = \
                self.stability_est_stabilizer.stabilize(approx_stability_limit)
            self.stability_est_stabilizer.update(pre_stabilization)
        adjustment_factor = self.stability_adjustment_adapter.get_current_stepsize()
        stepsize_upper_limit = adjustment_factor * approx_stability_limit
        dt = np.random.uniform(.5, 1) * stepsize_upper_limit

        # Sample the coefficients.
        beta_precond = beta / precond_scale
        f = self.get_precond_logprob_and_gradient(
            model, precond_scale, precond_prior_prec
        )
        if method == 'hmc':
            integration_time = np.pi / 2 * np.random.uniform(.8, 1.)
            n_step = np.ceil(integration_time / dt).astype('int')
            n_step = min(n_step, max_step)
            beta_precond, hmc_info = \
                hmc.generate_next_state(f, dt, n_step, beta_precond)
            info = {
                key: hmc_info.get(key)
                for key in ['accepted', 'accept_prob', 'n_grad_evals',
                            'instability_detected']
            }
            info['n_integrator_step'] = n_step
            hamiltonian_error = hmc_info['hamiltonian_error']

        elif method == 'nuts':
            nuts = NoUTurnSampler(f)
            max_height = int(math.log2(max_step))
            beta_precond, nuts_info = \
                nuts.generate_next_state(dt, beta_precond, max_height=max_height)
            info = {
                key: nuts_info.get(key)
                for key in ['ave_accept_prob', 'n_grad_evals', 'tree_height',
                            'instability_detected']
            }
            hamiltonian_error = nuts_info['ave_hamiltonian_error']

        else:
            raise NotImplementedError()

        beta = beta_precond * precond_scale
        self.regcoef_summarizer.update(beta, gscale, lscale)
        self.stability_adjustment_adapter.adapt_stepsize(hamiltonian_error)

        info['n_hessian_matvec'] = n_hessian_matvec
        info['stepsize'] = dt
        info['stability_limit_est'] = approx_stability_limit
        info['stability_adjustment_factor'] = adjustment_factor
        return beta, info

    def compute_preconditioning_scale(
            self, gscale, lscale, regcoef_precond_post_sd,
            prior_sd_for_unshrunk, unshrunk_target_sd_scale=1.):
        n_coef = len(regcoef_precond_post_sd)
        n_unshrunk = n_coef - len(lscale)

        precond_scale = np.ones(n_coef)
        precond_scale[n_unshrunk:] \
            = self.compute_prior_shrunk_scale(gscale, lscale)
        if n_unshrunk > 0:
            precond_scale[:n_unshrunk] = \
                unshrunk_target_sd_scale * regcoef_precond_post_sd[:n_unshrunk]

        precond_prior_prec = np.concatenate((
            (prior_sd_for_unshrunk / precond_scale[:n_unshrunk]) ** -2,
            np.ones(len(lscale))
        ))

        return precond_scale, precond_prior_prec

    def compute_prior_shrunk_scale(self, gscale, lscale):
        """ Compute the prior scale for the coefficient under regularized
        shrinkage in a numerically stable way. """
        prior_shrunk_scale = gscale * lscale
        prior_shrunk_scale /= np.sqrt(
            1 + (prior_shrunk_scale / self.regularizing_slab_size) ** 2
        )
        return prior_shrunk_scale

    def compute_stability_limit(
            self, gscale, lscale, model, precond_scale, precond_prior_prec):
        beta_condmean_guess = \
            self.regcoef_summarizer.extrapolate_beta_condmean(gscale, lscale)
        hessian_pc_estimate = self.regcoef_summarizer.estimate_precond_hessian_pc()
        max_curvature, hessian_pc, n_hessian_matvec = \
            self.compute_precond_hessian_curvature(
                beta_condmean_guess, model, precond_scale, precond_prior_prec,
                hessian_pc_estimate
            )
        self.regcoef_summarizer.update_precond_hessian_pc(hessian_pc)
        approx_stability_limit = 2 / np.sqrt(max_curvature)
        return approx_stability_limit, n_hessian_matvec

    def compute_precond_hessian_curvature(
            self, beta_location, model, precond_scale, precond_prior_prec, pc_estimate):

        precond_hessian_matvec, matvec_counter = self.get_precond_hessian_matvec(
            model, beta_location, precond_scale, precond_prior_prec
        )
        precond_hessian_op = sp.sparse.linalg.LinearOperator(
            (len(beta_location), len(beta_location)), precond_hessian_matvec
        )
        if pc_estimate is None:
            pc_estimate = np.random.randn(len(beta_location))
        eigval, eigvec = sp.sparse.linalg.eigsh(
            precond_hessian_op, k=1, tol=.1, v0=pc_estimate, ncv=2
        )   # We don't need a high (relative) accuracy.
        max_curvature = eigval[0]
        pc = np.squeeze(eigvec)
        if max_curvature <= 0:
            raise ArithmeticError(
                "Numerical instability occured during the Lancoz iteration and "
                "a negative curvature value returned for a log-concave distribution. "
                "Likely caused by divergence of regression coefficients to infinity. "
                "Check the input data and / or place an informative prior."
            )
        return max_curvature, pc, matvec_counter[0]

    @staticmethod
    def get_precond_hessian_matvec(
            model, beta_location, precond_scale, precond_prior_prec, obs_prec=None):

        if model.name == 'linear':
            args = [beta_location, obs_prec]
        else:
            args = [beta_location]
        loglik_hessian_matvec = model.get_hessian_matvec_operator(*args)
        matvec_counter = [0]
        def precond_hessian_matvec(beta_precond):
            matvec_counter[0] += 1
            return precond_prior_prec * beta_precond \
                   - precond_scale * loglik_hessian_matvec(precond_scale * beta_precond)

        return precond_hessian_matvec, matvec_counter

    @staticmethod
    def get_precond_logprob_and_gradient(
            model, precond_scale, precond_prior_prec, obs_prec=None):

        compute_loglik_and_gradient = model.compute_loglik_and_gradient
        if model.name == 'linear':
            compute_loglik_and_gradient \
                = partial(compute_loglik_and_gradient, obs_prec=obs_prec)
        def f(beta_precond, loglik_only=False):
            beta = beta_precond * precond_scale
            logp, grad_wrt_beta = \
                compute_loglik_and_gradient(beta, loglik_only=loglik_only)
            logp += np.sum(- precond_prior_prec * beta_precond ** 2) / 2

            grad = None
            if not loglik_only and math.isfinite(logp):
                grad = precond_scale * grad_wrt_beta  # Chain rule.
                grad += - precond_prior_prec * beta_precond
            return logp, grad

        return f

    def search_mode(self, beta, lscale, gscale, obs_prec, model, optim_maxiter=None,
                    use_newton_method=False, require_trust_region=False,
                    warn_optim_failure=False):

        if (not use_newton_method) and require_trust_region:
            warn("Trust regions are used only for Newton methods.")

        precond_scale, compute_negative_logp, compute_negative_grad, precond_hessian_matvec \
            = self.define_function_for_optim(beta, lscale, gscale, obs_prec, model)
        if not use_newton_method:
            precond_hessian_matvec = None

        optim_method, optim_options = self.choose_optim_method_and_options(
            optim_maxiter, use_newton_method, require_trust_region, n_param=len(beta)
        )

        beta_precond = beta / precond_scale
        model.design.memoize_dot(True)
        model.design.reset_matvec_count()
            # Avoid matrix-vector multiplication with the same input.
        optim_result = sp.optimize.minimize(
            compute_negative_logp, beta_precond, method=optim_method,
            jac=compute_negative_grad, hessp=precond_hessian_matvec,
            options=optim_options
        )
        model.design.memoize_dot(False)
        if (not optim_result.success) and warn_optim_failure:
            warn(
                "The regression coefficient mode (conditionally on the scale "
                "parameters could not be located within {:d} iterations of optimization "
                "steps. Proceeding with the current best estimate.".format(
                    optim_result.nit
                )
            )
        beta = precond_scale * optim_result.x
        info = {
            'is_success': optim_result.success,
            'method': optim_method,
            'n_iter': optim_result['nit'],
            'n_logp_eval': optim_result['nfev'],
            'n_grad_eval': optim_result.get('njev', 0),
                # incorrect output as of the current Scipy version (to be fixed in ver. 1.3.0)
            'n_hess_eval': optim_result.get('nhev', 0),
                # incorrect output as of the current Scipy version (to be fixed in ver. 1.3.0)
            'n_design_matvec': model.design.n_matvec,
        }
        return beta, info

    def define_function_for_optim(self, beta, lscale, gscale, obs_prec, model):

        beta_precond_post_sd = np.ones(beta.size)
        # No Monte Carlo estimate yet, so make some reasonable guess. It
        # probably should depend on the outcome and design matrix.
        precond_scale, precond_prior_prec = \
            self.compute_preconditioning_scale(
                gscale, lscale, beta_precond_post_sd, self.prior_sd_for_unshrunk
            )

        f = self.get_precond_logprob_and_gradient(
            model, precond_scale, precond_prior_prec, obs_prec
        )
        def compute_negative_logp(beta_precond):
            # Negative log-density
            return - f(beta_precond, loglik_only=True)[0]

        def compute_negative_grad(beta_precond):
            return - f(beta_precond)[1]


        def precond_hessian_matvec(precond_location, v):
            hessian_eval_location = precond_scale * precond_location
            hessian_matvec, _ = self.get_precond_hessian_matvec(
                model, hessian_eval_location, precond_scale, precond_prior_prec,
                obs_prec=obs_prec
            )
            return hessian_matvec(v)

        return precond_scale, compute_negative_logp, compute_negative_grad, precond_hessian_matvec

    def choose_optim_method_and_options(
            self, optim_maxiter, use_newton_method, require_trust_region, n_param):

        if optim_maxiter is None:
            if use_newton_method:
                optim_maxiter = 15
            else:
                optim_maxiter = 250

        optim_options = {'maxiter': optim_maxiter}
        tol = 10 ** -6 / np.sqrt(n_param)  # In analogy with the CG-sampler.

        if not use_newton_method:
            optim_method = 'L-BFGS-B'
            optim_options['gtol'] = tol
            optim_options['maxcor'] = 200
        else:
            if require_trust_region:
                optim_method = 'trust-ncg'
                # Start with a generous trust radius as Newton iterations without
                # constraints should be fine.
                init_trust_radius = 1.96 * np.sqrt(n_param)
                optim_options.update({
                    'gtol': tol,
                    'initial_trust_radius': init_trust_radius,
                    'max_trust_radius': 4. * init_trust_radius
                })
            else:
                optim_method = 'Newton-CG'
                optim_options['xtol'] = tol

        return optim_method, optim_options


class StabilityEstimateStabilizer():
    """ Detect and adjust unusually large stability limit estimates. """

    def __init__(self, n_warmup=100):
        self.stability_estimate = []
        self.n_update = 0
        self.n_warmup = n_warmup

    def update(self, estimate):
        self.stability_estimate.append(estimate)
        self.n_update += 1

    def stabilize(self, estimate):

        if self.n_update < self.n_warmup:
            return estimate

        gaussian_cdf_at_onestd = .8414
        stability_estimate = np.array(self.stability_estimate[:self.n_update])
        cdf_at_estimate = np.mean(stability_estimate < estimate)

        if cdf_at_estimate <= gaussian_cdf_at_onestd:
            stabilized_est = estimate
        else:
            stability_median = np.median(stability_estimate)
            stability_at_gaussian_cdf = np.quantile(
                stability_estimate, gaussian_cdf_at_onestd
            )
            one_std_dist = stability_at_gaussian_cdf - stability_median
            gaussian_dist_above_onestd = sp.stats.norm.ppf(cdf_at_estimate) - 1.
            gaussian_dist_above_onestd = min(2., gaussian_dist_above_onestd)
                # In case the empirical CDF of the estimate is too close to 1.
            stabilized_est = stability_at_gaussian_cdf \
                    + one_std_dist * gaussian_dist_above_onestd

        return stabilized_est
