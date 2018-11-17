import numpy as np

def numerical_grad_is_close(f, x, atol=10E-6, rtol=10E-6, dx=10E-6):
    """
    Compare the computed gradient to a centered finite difference approximation.

    Params:
    -------
    f : callable
        Returns a value of a function and its gradient
    """
    x = np.array(x, ndmin=1)
    grad_est = np.zeros(len(x))
    for i in range(len(x)):
        x_minus = x.copy()
        x_minus[i] -= dx
        x_plus = x.copy()
        x_plus[i] += dx
        f_minus, _ = f(x_minus)
        f_plus, _ = f(x_plus)
        grad_est[i] = (f_plus - f_minus) / (2 * dx)

    _, grad = f(x)
    return np.allclose(grad, grad_est, atol=atol, rtol=rtol)


def numerical_direc_deriv_is_close(
        f, x, hess_matvec, n_direction=10,
        atol=10E-6, rtol=10E-6, dx=10E-6, seed=None):
    """
    Compare analytically computed directional derivatives of the gradient of 'f'
    (i.e. the Hessian of 'f' applied to vectors) to its numerical approximations.

    Params:
    -------
    f : callable
        Returns a value of a function and its gradient
    """

    x = np.array(x, ndmin=1)

    np.random.seed(seed)
    all_matched = True

    for i in range(n_direction):

        v = np.random.randn(len(x))
        v /= np.sqrt(np.sum(v ** 2))
        _, grad_minus = f(x - dx * v)
        _, grad_plus = f(x + dx * v)
        direc_deriv_est = (grad_plus - grad_minus) / (2 * dx)
        direc_deriv = hess_matvec(v)

        if not np.allclose(direc_deriv, direc_deriv_est, atol=atol, rtol=rtol):
            all_matched = False
            break

    return all_matched
