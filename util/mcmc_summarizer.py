import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import MaxNLocator


def plot_conf_interval(
        coef_samples, conf_level=.95, n_coef_to_plot=None,
        sort_by_median_val=False, marker_scale=1.0,
    ):
    tail_prob = (1 - conf_level) / 2
    lower, median, upper = [
        np.quantile(coef_samples, q, axis=-1)
        for q in [tail_prob, .5, 1 - tail_prob]
    ]

    if sort_by_median_val:
        sort_ind = np.argsort(median)
    else:
        sort_ind = np.arange(len(median))  # No sorting

    if n_coef_to_plot is None:
        n_coef_to_plot = len(median)
    coef_index = sort_ind[:n_coef_to_plot]

    plt.plot(
        coef_index, median[coef_index],
        'x', color='tab:blue', ms=marker_scale * 10,
        label='Posterior median'
    )
    plt.plot(
        coef_index, lower[coef_index],
        '_', color='tab:green', ms=marker_scale * 12, lw=marker_scale * 1.2,
        label='{:.1f}% credible interval'.format(100 * conf_level)
    )
    plt.plot(
        coef_index, upper[coef_index],
        '_', color='tab:green', ms=marker_scale * 12, lw=marker_scale * 1.2
    )
    plt.gca().get_xaxis().set_major_locator(MaxNLocator(integer=True))

    plotted_quantity = {
        'lower': lower[coef_index],
        'median': median[coef_index],
        'upper': upper[coef_index],
        'coef_index': coef_index
    }
    return plotted_quantity