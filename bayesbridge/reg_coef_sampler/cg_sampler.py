import numpy as np
import scipy as sp
import scipy.sparse
import scipy.linalg
from warnings import warn

try:
    import cupyx.scipy.sparse.linalg
    import cupyx as cpx
    import cupy as cp
except (ImportError, ModuleNotFoundError) as e:
    cp = None
    cupy_exception = e

class ConjugateGradientSampler():

    def __init__(self, n_coef_wo_shrinkage):
        self.n_coef_wo_shrinkage = n_coef_wo_shrinkage

    def sample(
            self, X, omega, prior_prec_sqrt, z,
            beta_init=None, precond_by='prior', beta_scaled_sd=None,
            maxiter=None, atol=10e-6, seed=None):
        """
        Generate a multi-variate Gaussian with the mean mu and covariance Sigma of the form
           Sigma^{-1} = X' Omega X + prior_prec_sqrt^2, mu = Sigma z
        where D is assumed to be diagonal. For numerical stability, the code first sample
        from the scaled parameter regress_coef / precond_scale.
        Param:
        ------
        D : vector
        atol : float
            The absolute tolerance on the residual norm at the termination
            of CG iterations.
        beta_scaled_sd : vector of length X.shape[1]
            Used to estimate a good preconditioning scale for the coefficient
            without shrinkage. Used only if precond_by == 'prior'.
        precond_by : {'prior', 'diag'}
        """
        if X.use_cupy:
            beta_init = cp.asarray(beta_init)
            beta_scaled_sd = cp.asarray(beta_scaled_sd)
            prior_prec_sqrt = cp.asarray(prior_prec_sqrt)
            cg = cpx.scipy.sparse.linalg.cg
            LinearOperator = cpx.scipy.sparse.linalg.LinearOperator
        else:
            cg = sp.sparse.linalg.cg
            LinearOperator = sp.sparse.linalg.LinearOperator
        if seed is not None:
            np.random.seed(seed)

        # Define a preconditioned linear operator.
        Phi_precond_op, precond_scale = \
            self.precondition_linear_system(
                prior_prec_sqrt, omega, X, precond_by, beta_scaled_sd, LinearOperator
            )

        # Draw a target vector.
        randn_vec_1 = np.random.randn(X.shape[0])
        randn_vec_2 = np.random.randn(X.shape[1])
        if X.use_cupy:
            randn_vec_1 = cp.asarray(randn_vec_1)
            randn_vec_2 = cp.asarray(randn_vec_2)
        v = X.Tdot(omega ** (1 / 2) * randn_vec_1) \
            + prior_prec_sqrt * randn_vec_2
        b = precond_scale * (z + v)

        # Callback function to count the number of PCG iterations.
        cg_info = {'n_iter': 0}
        def cg_callback(x): cg_info['n_iter'] += 1

        # Run PCG.
        rtol = atol / np.linalg.norm(b)
        beta_scaled_init = beta_init / precond_scale
        beta_scaled, info = cg(
            Phi_precond_op, b, x0=beta_scaled_init, maxiter=maxiter, tol=rtol,
            callback=cg_callback
        )

        if info != 0:
            warn(
                "The conjugate gradient algorithm did not achieve the requested " +
                "tolerance level. You may increase the maxiter or use the dense " +
                "linear algebra instead."
            )

        beta = precond_scale * beta_scaled
        cg_info['valid_input'] = (info >= 0)
        cg_info['converged'] = (info == 0)
        if X.use_cupy:
            beta = cp.asnumpy(beta)
        return beta, cg_info

    def precondition_linear_system(
            self, prior_prec_sqrt, omega, X, precond_by, beta_scaled_sd, LinearOperator):

        # Compute the preconditioners.
        precond_scale = self.choose_preconditioner(
            prior_prec_sqrt, omega, X, precond_by, beta_scaled_sd
        )

        # Define a preconditioned linear operator.
        precond_prior_prec = (precond_scale * prior_prec_sqrt) ** 2
        def Phi_precond(x):
            Phi_x = precond_prior_prec * x \
                    + precond_scale * X.Tdot(omega * X.dot(precond_scale * x))
            return Phi_x
        Phi_precond_op = LinearOperator(
            (X.shape[1], X.shape[1]), matvec=Phi_precond
        )
        return Phi_precond_op, precond_scale

    def choose_preconditioner(
            self, prior_prec_sqrt, omega, X, precond_by, beta_scaled_sd):

        precond_scale = self.choose_diag_preconditioner(
            prior_prec_sqrt, omega, X, precond_by, beta_scaled_sd)

        return precond_scale

    def choose_diag_preconditioner(
            self, prior_prec_sqrt, omega, X, precond_by='diag',
            beta_scaled_sd=None):
        # Compute the diagonal (sqrt) preconditioner.

        if precond_by == 'prior':
            precond_scale = cp.ones(len(prior_prec_sqrt)) if X.use_cupy \
                else np.ones(len(prior_prec_sqrt))
            precond_scale[self.n_coef_wo_shrinkage:] = \
                prior_prec_sqrt[self.n_coef_wo_shrinkage:] ** -1
            if self.n_coef_wo_shrinkage > 0:
                target_sd_scale = 2.
                    # Larger than 1 because it is better to err on the side
                    # of introducing large precisions.
                precond_scale[:self.n_coef_wo_shrinkage] = \
                    target_sd_scale * beta_scaled_sd[:self.n_coef_wo_shrinkage]

        elif precond_by == 'diag':
            diag = prior_prec_sqrt ** 2 + X.compute_fisher_info(weight=omega, diag_only=True)
            precond_scale = 1 / np.sqrt(diag)

        elif precond_by is None:
            precond_scale = np.ones(X.shape[1])

        else:
            raise NotImplementedError()

        return precond_scale