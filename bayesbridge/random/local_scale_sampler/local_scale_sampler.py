import math
from random import random as rand_unif
from random import expovariate as rand_exp
# from .tilted_stable_dist.custom_math import exp, log
import numpy as np


def sample_horseshoe_local_scale(beta_shrunk, gshrink):
    b = .5 * (beta_shrunk / gshrink) ** 2
    b[b == 0] = 2. ** -1023
    b[np.isinf(b)] = 2. ** 1023
    local_prec_transformed = np.array([
        sample_from_transformed_precision(b_j) for b_j in b
    ])
    local_prec = np.exp(local_prec_transformed) - 1.
    return 1 / np.sqrt(local_prec)


def sample_from_transformed_precision(b):
    """
    Sample from a density proportional to exp(- b * exp(x)) by
    rejection sampler.
    """
    accepted = False
    while not accepted:
        rv = sample_from_bounding_dist(b)
        target_logp = - b * np.exp(rv)
        bound_logp = compute_bounding_logp(rv, b)
        accept_prob = np.exp(target_logp - bound_logp)
        accepted = accept_prob > rand_unif()
    return rv


def sample_from_bounding_dist(b):
    if b < 1.:
        rv = sample_from_mixture(b)
    else:
        rv = rand_exp(1.) / b
    return rv


def sample_from_mixture(b):
    """ Sample from a mixture of Unif(0, - log(b)) and - log(b) + Exp(1). """
    log_b = np.log(b)
    mass_to_left = np.exp(-b) * (- log_b)
    mass_to_right = 0.36787944117144233  # = exp(-1)
    prob_to_left = mass_to_left / (mass_to_left + mass_to_right)
    sampled_from_left = prob_to_left > rand_unif()
    if sampled_from_left:
        rv = (- log_b) * rand_unif()
    else:
        rv = - log_b + rand_exp(1.)
    return rv


def compute_bounding_logp(x, b):
    if b < 1:
        thresh = - np.log(b)
        if x < thresh:
            logp = - b
        else:
            logp = - 1 - (x - thresh)
    else:
        logp = - b * (1 + x)
    return logp