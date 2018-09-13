import numpy as np
import scipy as sp
import scipy.sparse
import scipy.linalg
from .sparse_dense_matrix_operators \
    import elemwise_power, left_matmul_by_diag, right_matmul_by_diag, \
    choose_optimal_format_for_matvec
from util.simple_warnings import warn_message_only

class ConjugateGradientSampler():

    def __init__(self, n_coef_wo_shrinkage):
        self.n_coef_wo_shrinkage = n_coef_wo_shrinkage

    def sample(
            self, X_row_major, X_col_major, omega, prior_prec_sqrt, z,
            beta_init_1=None, beta_init_2=None,
            precond_by='diag', precond_blocksize=0, beta_scaled_sd=None,
            maxiter=None, atol=10e-6, seed=None):
        """
        Generate a multi-variate Gaussian with the mean mu and covariance Sigma of the form
           Sigma^{-1} = X' Omega X + prior_prec_sqrt^2, mu = Sigma z
        where D is assumed to be diagonal. For numerical stability, the code first sample
        from the scaled parameter beta / precond_scale.

        Param:
        ------
        prior_prec_sqrt : vector
        atol : float
            The absolute tolerance on the residual norm at the termination
            of CG iterations.
        beta_scaled_sd : vector of length X.shape[1]
            Used to estimate a good preconditioning scale for the coefficient
            without shrinkage. Used only if precond_by in ('prior', 'prior+block').
        """

        X, X_T = choose_optimal_format_for_matvec(X_row_major, X_col_major)

        if seed is not None:
            np.random.seed(seed)

        # Define a preconditioned linear operator.
        Phi_precond_op, precond_scale, block_precond_op = \
            self.precondition_linear_system(
                prior_prec_sqrt, omega, X_row_major, X_col_major, precond_by,
                precond_blocksize, beta_scaled_sd
            )

        # Draw a target vector.
        v = X_T.dot(omega ** (1 / 2) * np.random.randn(X.shape[0])) \
            + prior_prec_sqrt * np.random.randn(X.shape[1])
        b = precond_scale * (z + v)

        # Pick the initial vector for CG iteration
        beta_scaled_init = self.choose_best_linear_comb(
            beta_init_1, beta_init_2, Phi_precond_op, precond_scale, b
        )

        # Callback function to count the number of PCG iterations.
        cg_info = {'n_iter': 0}
        def cg_callback(x): cg_info['n_iter'] += 1

        # Run PCG.
        rtol = atol / np.linalg.norm(b)
        beta_scaled, info = sp.sparse.linalg.cg(
            Phi_precond_op, b, x0=beta_scaled_init, maxiter=maxiter, tol=rtol,
            M=block_precond_op, callback=cg_callback
        )

        if info != 0:
            warn_message_only(
                "The conjugate gradient algorithm did not achieve the requested " +
                "tolerance level. You may increase the maxiter or use the dense " +
                "linear algebra instead."
            )

        beta = precond_scale * beta_scaled
        cg_info['valid_input'] = (info >= 0)
        cg_info['converged'] = (info == 0)

        return beta, cg_info

    def precondition_linear_system(
            self, D, omega, X_row_major, X_col_major, precond_by,
            precond_blocksize, beta_scaled_sd):

        X, X_T = choose_optimal_format_for_matvec(X_row_major, X_col_major)

        # Compute the preconditioners.
        precond_scale, block_precond_op = self.choose_preconditioner(
            D, omega, X_row_major, X_col_major, precond_by, precond_blocksize, beta_scaled_sd
        )

        # Define a preconditioned linear operator.
        D_scaled_sq = (precond_scale * D) ** 2
        def Phi_precond(x):
            Phi_x = D_scaled_sq * x \
                    + precond_scale * X_T.dot(omega * X.dot(precond_scale * x))
            return Phi_x
        Phi_precond_op = sp.sparse.linalg.LinearOperator(
            (X.shape[1], X.shape[1]), matvec=Phi_precond
        )
        return Phi_precond_op, precond_scale, block_precond_op

    def choose_preconditioner(self, D, omega, X_row_major, X_col_major,
                              precond_by, precond_blocksize, beta_scaled_sd):

        precond_scale = self.choose_diag_preconditioner(
            D, omega, X_row_major, precond_by, beta_scaled_sd)

        block_precond_op = None
        if precond_by == 'prior+block' and precond_blocksize > 0:
            pred_importance = precond_scale.copy()
            pred_importance[:self.n_coef_wo_shrinkage] = float('inf')
            subset_indices = np.argsort(pred_importance)[-precond_blocksize:]
            subset_indices = np.sort(subset_indices)
            block_precond_op = self.compute_block_preconditioner(
                omega, X_row_major, X_col_major, D, precond_scale, subset_indices
            )

        return precond_scale, block_precond_op

    def choose_diag_preconditioner(
            self, D, omega, X_row_major, precond_by='diag',
            beta_scaled_sd=None):
        # Compute the diagonal (sqrt) preconditioner.

        if precond_by in ('prior', 'prior+block'):

            precond_scale = D ** -1
            if self.n_coef_wo_shrinkage > 0:
                target_sd_scale = 2.
                    # Larger than 1 because it is better to err on the side
                    # of introducing large precisions.
                precond_scale[:self.n_coef_wo_shrinkage] = \
                    target_sd_scale * beta_scaled_sd[:self.n_coef_wo_shrinkage]

        elif precond_by == 'diag':
            diag = D ** 2 + np.squeeze(np.asarray(
                left_matmul_by_diag(
                    omega, elemwise_power(X_row_major, 2)
                ).sum(axis=0)
            ))
            precond_scale = 1 / np.sqrt(diag)

        elif precond_by is None:
            precond_scale = np.ones(X_row_major.shape[1])

        else:
            raise NotImplementedError()

        return precond_scale

    def compute_block_preconditioner(
            self, omega, X_row_major, X_col_major, D, precond_scale, indices):

        if X_col_major is not None:
            X = X_col_major
        else:
            X = X_row_major

        weighted_X_subset = \
            left_matmul_by_diag(omega ** (1 / 2), X[:, indices])
        weighted_X_subset_scaled = right_matmul_by_diag(weighted_X_subset, precond_scale[indices])
        if sp.sparse.issparse(weighted_X_subset_scaled):
            weighted_X_subset_scaled = weighted_X_subset_scaled.toarray()
        B = weighted_X_subset_scaled.T.dot(weighted_X_subset_scaled) \
            + np.diag((D[indices] * precond_scale[indices]) ** 2)

        B_cho_factor = sp.linalg.cho_factor(B)
        def B_inv_on_indices(x):
            x = x.copy() # TODO: Check if a shallow copy is OK.
            x[indices] = sp.linalg.cho_solve(B_cho_factor, x[indices])
            return x
        block_preconditioner_op = sp.sparse.linalg.LinearOperator(
            (X.shape[1], X.shape[1]), matvec=B_inv_on_indices
        )

        return block_preconditioner_op

    def choose_best_linear_comb(
            self, beta_init_1, beta_init_2, Phi_precond_op, precond_scale, b):

        if beta_init_1 is not None:
            beta_init_1 = beta_init_1.copy() / precond_scale
        if beta_init_2 is not None:
            beta_init_2 = beta_init_2.copy() / precond_scale
        beta_scaled_init = self.optimize_cg_objective(
            Phi_precond_op, b, beta_init_1, beta_init_2)

        return beta_scaled_init

    def optimize_cg_objective(self, A, b, x1, x2=None):
        # Minimize the function f(x) = x'Ax / 2 - x'b along the line connecting
        # x1 and x2.
        if x2 is None:
            x = x1
        else:
            v = x2 - x1
            Av = A(v)
            denom = v.dot(Av)
            if denom == 0:
                t_argmin = 0
            else:
                t_argmin = (x1.dot(Av) - b.dot(v)) / denom
            x = x1 - t_argmin * v
        return x