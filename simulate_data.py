import numpy as np
import scipy as sp
import scipy.sparse
from bayesbridge.model import CoxModel


def simulate_outcome(X, beta, model, intercept=0., n_trial=None, seed=None):

    if seed is not None:
        np.random.seed(seed)

    if model == 'linear':
        sigma = 1.
        outcome = intercept + X.dot(beta) + sigma * np.random.randn(X.shape[0])
    elif model == 'logit':
        if n_trial is None:
            n_trial = np.ones(X.shape[0])
        prob = 1 / (1 + np.exp(- intercept - X.dot(beta)))
        n_success = np.random.binomial(n_trial.astype(np.int64), prob)
        outcome = (n_success, n_trial)
    elif model == 'cox':
        outcome = CoxModel.simulate_outcome(X, beta, censoring_frac=.5)
    else:
        raise NotImplementedError()

    return outcome

def simulate_design(
        n_obs, n_pred, binary_frac=0., categorical_frac=0.,
        corr_dense_design=False, binary_pred_freq=.1, n_category=5,
        shuffle_columns=False, seed=None, format_='sparse'
    ):

    if seed is not None:
        np.random.seed(seed)

    n_dense_pred = int(n_pred * (1 - binary_frac - categorical_frac))
    n_categorical_pred = int((n_pred * categorical_frac) / (n_category - 1))
    n_binary_pred = n_pred - n_dense_pred - n_categorical_pred * (n_category - 1)

    X_dense = simulate_dense_design(n_obs, n_dense_pred, corr_dense_design)

    if n_binary_pred + n_categorical_pred == 0:
        X = X_dense
    else:
        X_binary = simulate_binary_design(n_obs, n_binary_pred, binary_pred_freq)
        X_categorical = simulate_categorical_design(
            n_obs, n_categorical_pred, n_category
        )
        X = sp.sparse.hstack((
            sp.sparse.csr_matrix(X_dense), X_binary, X_categorical
        )).tocsr()

    if shuffle_columns:
        X = X[:, np.random.permutation(n_pred)]

    if format_ == 'sparse':
        X = sp.sparse.csr_matrix(X)
    elif sp.sparse.issparse(X):
        X = X.toarray()

    return X

def simulate_dense_design(n_obs, n_pred, corr_design, standardize=False):
    if corr_design:
        X = generate_corr_design(n_obs, n_pred)
    else:
        X = np.random.randn(n_obs, n_pred)
    if standardize:
        X = np_standardize(X)
    return X

def np_standardize(X, divide_by='std'):
    X = X - np.mean(X, axis=0)[np.newaxis, :]
    if divide_by == 'max':
        X = X / np.max(X, axis=0)[np.newaxis, :]
    else:
        X = X / np.std(X, axis=0)[np.newaxis, :]
    return X

def generate_corr_design(n_obs, n_pred, n_factor=None, max_sd=100, min_sd=1):
    """
    Each column is drawn from a Gaussian with a covariance proportional to
        I + F L F'
    where F is an orthogonal matrix of size p by n_factor and L is diagonal.
    """
    if n_factor is None:
        n_factor = min(100, int(n_pred / 2))
    factor, _ = np.linalg.qr(np.random.randn(n_pred, n_factor))
    principal_comp_sd = np.linspace(max_sd, min_sd, n_factor + 1)
    loading = principal_comp_sd[:n_factor] - min_sd
    X = np.dot(
        factor,
        loading[:, np.newaxis] * np.random.randn(n_factor, n_obs)
    ).T
    X += min_sd * np.random.randn(n_obs, n_pred)
    return X

def simulate_binary_design(n_obs, n_binary_pred, sparsity, max_freq_per_col=.5):
    """
    Returns a binary matrix where the non-zero frequency (on average) equals
    the value of 'sparsity'. Also, the non-zero frequency along each column is
    bounded by 'max_freq_per_col'.
    """
    if n_binary_pred == 0:
        return None

    a = .5
    b = a * (max_freq_per_col / sparsity - 1)
        # Solve a / (a + b) = sparsity / max_freq_per_col for 'b'.
    binary_freq = max_freq_per_col * np.random.beta(a, b, n_binary_pred)
    X = np.random.binomial(1, binary_freq, (n_obs, n_binary_pred))
    return X

def simulate_categorical_design(n_obs, n_categorical_pred, n_category=5):
    if n_categorical_pred == 0:
        return None

    X = sp.sparse.hstack([
        sp.sparse.csr_matrix(draw_categorical_pred(n_obs, n_category))
        for dummy in range(n_categorical_pred)
    ])
    return X

def draw_categorical_pred(n_obs, n_category):
    # Returns a matrix of size n by (n_category - 1).
    category_freq = np.random.dirichlet(
        np.ones(n_category - 1) / n_category
    )
    return np.random.multinomial(1, category_freq, n_obs)
