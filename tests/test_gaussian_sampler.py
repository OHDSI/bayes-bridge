import numpy as np

from bayesbridge.design_matrix import SparseDesignMatrix, DenseDesignMatrix
from bayesbridge.reg_coef_sampler.direct_gaussian_sampler \
    import matvec_by_post_prec_inverse_via_woodbury
from simulate_data import simulate_design


def test_matvec_via_woodbury():
    np.random.seed(0)
    n_obs, n_pred = (5, 3)
    X = simulate_design(n_obs, n_pred, binary_frac=.5, format_='sparse')
    design = SparseDesignMatrix(
        X, center_predictor=True, add_intercept=True
    )
    prior_prec_sqrt = np.random.exponential(size=n_pred + 1)
    obs_prec = np.random.exponential(size=n_obs)
    Post_prec = \
        np.diag(prior_prec_sqrt ** 2) \
        + design.compute_fisher_info(weight=obs_prec)
    x = np.random.randn(n_pred + 1)
    wb_solution = matvec_by_post_prec_inverse_via_woodbury(
        design, obs_prec, prior_prec_sqrt, x
    )
    assert np.allclose(
        wb_solution, np.linalg.solve(Post_prec, x),
        atol=10e-6, rtol=10e-6
    )

