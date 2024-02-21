import sys
# sys.path.append('.')
import numpy as np
import matplotlib.pyplot as plt


def log_prob_c_pos(x, a, c):
    return - a ** 2 * (np.sqrt(np.exp(2 * x) - 1) - c) ** 2


def log_prob_c_neg(x, a, c):
    eta = np.exp(2 * x) - 1
    return - a ** 2 * eta + 2 * a ** 2 * c * np.sqrt(eta)


def log_prob(x, a, c):
    if c >= 0:
        return log_prob_c_pos(x, a, c)
    else:
        return log_prob_c_neg(x, a, c)


def compute_target_pdf(x, a, c, normalized=True):
    log_p = log_prob(x, a, c)
    log_p -= np.max(log_p)  # Avoid numerical under-flow when exponentiation.
    prob = np.exp(log_p)
    if normalized:
        prob = prob / np.trapz(y=prob, x=x)
    return prob


def plot_hist_against_target(ax, lscale_samples, a, c):
    # Restrict the plot range; otherwise, the empirical
    # distribution of a heavy-tailed target is unstable.
    max_quantile = .99
    upper_lim = np.quantile(lscale_samples, max_quantile)
    bins = np.linspace(0, upper_lim, 51)

    x = np.linspace(0, upper_lim, 1001)[1:]
    ax.hist(lscale_samples, bins=bins, density=True,
            label='empirical dist')
    ax.plot(x, compute_target_pdf(x, a, c),
            label='target pdf')
    ax.set_xlabel(
        'a= {:1g}, c = {:1g}'.format(a, c))
    ax.set_yticks([])


def remove_figure_box_edges(ax, sides=None):
    if sides is None:
        sides = ['left', 'right', 'top']
    for side in sides:
        ax.spines[side].set_visible(False)
