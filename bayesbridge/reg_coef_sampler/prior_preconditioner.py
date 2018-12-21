import numpy as np

def compute_preconditioning_scale(gshrink, lshrink, reg_coef_precond_post_sd):

    n_coef = len(reg_coef_precond_post_sd)
    n_unshrunk = n_coef - len(lshrink)
    precond_scale = np.ones(n_coef)
    precond_scale[n_unshrunk:] = gshrink * lshrink
    if n_unshrunk > 0:
        target_sd_scale = 2.
        precond_scale[:n_unshrunk] = \
            target_sd_scale * reg_coef_precond_post_sd[:n_unshrunk]

    return precond_scale