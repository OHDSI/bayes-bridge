# TODO: find a way to compute/store mu_beta_a and var_beta_a,
#  one option is to store them in a csv file,
#  then for each beta, you find whether it's informed or not
import numpy as np
from ..random.local_scale_sampler import skewed_shrinkage_rejection_sampler


def compute_horseshoe_lscale(beta_coef,
                             gscale,
                             skew_prior_mean,
                             skew_prior_sd,
                             q=0.5, k1=16, k2=5):
    a = beta_coef / np.sqrt(2) * gscale * skew_prior_mean
    c = gscale * skew_prior_sd / beta_coef

    a_c_list = list(zip(a,c))
    lscale_inv_ar_list = [
        skewed_shrinkage_rejection_sampler(a_c[0], a_c[1], q=q, k1=k1, k2=k2)
        for a_c in a_c_list
    ]
    lscale_inv_arr = np.array([i[0] for i in lscale_inv_ar_list])
    acc_count = np.array([i[1] for i in lscale_inv_ar_list])
    return 1 / lscale_inv_arr, acc_count


def get_informed_prior(mu_beta_a, r, dist, sigma_sq_ave, sigma_sq_db):
    sigma_sq_coef = sigma_sq_ave + sigma_sq_db
    rho = np.exp(- r * dist)
    gamma = rho * sigma_sq_ave / sigma_sq_coef
    gamma_sq = gamma ** 2
    return gamma * mu_beta_a, np.sqrt((1 - gamma_sq) * sigma_sq_coef)
