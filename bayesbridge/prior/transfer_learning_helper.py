# TODO: find a way to compute/store mu_beta_a and var_beta_a,
#  one option is to store them in a csv file,
#  then for each beta, you find whether it's informed or not

def compute_horseshoe_lscale():
    if no_prior_info:
        lscale = sample_horseshoe_local_scale(coef, gscale)
    # TODO: Implemement the skewed version
    # the skewed version uses the horseshoe prior, with information from larger/another database

    # TODO: should be another function
    #  for each beta coefficient with prior information,
    #  do the following updates on the local scale param
    else:
        beta_mean_prior, beta_var_prior = get_informed_prior(mu_beta_A, var_beta_a, rho, sigma_sq_ave, sigma_sq_db)
        a = beta_coef / np.sqrt(2) * gscale * beta_sd_prior
        c = gscale_ * beta_mean_prior / beta_coef
        lscale = 1 / skewed_shrinkage_rejection_sampler(a, c, q=0.5, k1=16, k2=5)
    pass

def get_informed_prior(mu_beta_a, var_beta_a, rho, sigma_sq_ave, sigma_sq_db):
    sigma_sq_coef = sigma_sq_ave + sigma_sq_db
    gamma = rho * sigma_sq_ave / sigma_sq_coef
    gamma_sq = gamma**2
    return gamma * mu_beta_a, (1 - gamma_sq) * sigma_sq_coef + gamma_sq * var_beta_a