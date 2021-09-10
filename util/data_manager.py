import numpy as np
import scipy as sp
import pandas as pd
import pickle as pkl


class DataManager(object):

    def __init__(self, path_to_data, regress_on, n_strata=5, use_gi_outcome=False,
                 include_interaction=False, include_interaction_directly=False,
                 exclude_main_effect=False, center_treatment_only=False,
                 min_freq=None, use_relabeled=True):
        self.folder = path_to_data
        self.regress_on = regress_on
        self.n_strata = n_strata
        self.use_gi_outcome = use_gi_outcome
        self.include_interaction = include_interaction
        self.include_interaction_directly = include_interaction_directly
        self.exclude_main_effect = exclude_main_effect
        # Used only implicitly i.e. the main effects are not explicitly excluded.
        self.center_treatment_only = center_treatment_only
        if min_freq is None:
            min_freq = .01 if self.include_interaction else 0.
        self.min_freq = min_freq
        self.use_relabeled = use_relabeled

    def read_ohdsi_data(self):
        """ Read the outcome and design matrix. """

        X_bin = self.read_binary_design().tocsc()
        cov_freq = np.array(X_bin.mean(0)).ravel()
        if self.min_freq > 0:
            X_bin = X_bin[:, cov_freq > self.min_freq]
        X_cont = self.read_cont_design()
        X = sp.sparse.hstack((X_bin, X_cont))
        treatment = np.loadtxt(self.folder + 'treatment.txt')
        treatment = sp.sparse.csc_matrix(treatment[:, np.newaxis])

        if self.include_interaction_directly:
            X_treatment = self.create_interaction(treatment, X)
            X = sp.sparse.hstack((X, X_treatment))

        if self.regress_on == 'treatment':
            y = treatment
        elif self.regress_on == 'outcome':
            strata_id = self.read_propensity_strata(self.n_strata)
            X = sp.sparse.hstack((
                treatment, strata_id, X
            ))
            if self.use_gi_outcome:
                y = np.loadtxt(self.folder + 'gi_bleed.txt', dtype=np.int64)
            else:
                y = np.loadtxt(self.folder + 'outcome.txt', dtype=np.int64)
        elif self.regress_on == 'linearly_adjusted_outcome':
            X = sp.sparse.hstack((
                treatment, X
            ))
            y = np.loadtxt(self.folder + 'outcome.txt', dtype=np.int64)
        else:
            NotImplementedError()

        return y, X.tocsr()

    def read_binary_design(self):
        suffix = '_relabeled' if self.use_relabeled else ''
        finename = 'sparse_design_matrix_binary_part' + suffix + '.txt'
        return self.read_sp_coo(self.folder + finename)

    def read_cont_design(self):
        suffix = '_relabeled' if self.use_relabeled else ''
        finename = 'sparse_design_matrix_cont_part' + suffix + '.txt'
        return self.read_sp_coo(self.folder + finename)

    def read_sp_coo(self, filename):
        A = pd.read_table(filename, delim_whitespace=True,
                          dtype={'i': np.int32, 'j': np.int32, 'val': np.float64})
        A = sp.sparse.coo_matrix((A.val, (A.i, A.j)))
        return A

    def read_propensity_strata(self, n_strata):
        propensity_score = np.loadtxt(self.folder + 'propensity_score.txt')
        strata_ind = self.create_strata_indicator(propensity_score, n_strata)
        return strata_ind

    @staticmethod
    def create_strata_indicator(score, n_strata=5):
        percent = np.linspace(0, 100, n_strata + 1)
        percentile = np.percentile(score, percent)
        percentile[0] = -1  # Make sure the inequality behaves correctly.
        X = np.zeros(
            (len(score), n_strata - 1))  # -1 to avoid the collinearity
        for j in range(n_strata - 1):
            within_strata = np.logical_and(
                percentile[j] < score,
                score <= percentile[j + 1]
            )
            X[within_strata, j] = 1
        return sp.sparse.csr_matrix(X)

    @staticmethod
    def create_interaction(covariate, X):
        """ Create interaction between one covariate and columns of a design matrix. """
        X = X.tocsc()
        X_interaction = sp.sparse.hstack(tuple(
            X[:, j].multiply(covariate)
            for j in range(X.shape[1])
        ))
        return X_interaction

    def read_covariate_name(self):

        cont_covariate_name = np.loadtxt(
            self.folder + 'covariate_name_cont_part.txt',
            dtype=np.str_, delimiter='\n'
        )
        bin_convariate_name = np.loadtxt(
            self.folder + 'covariate_name_binary_part.txt',
            dtype=np.str_, delimiter='\n'
        )
        if self.include_interaction and self.min_freq > 0:
            X_bin = self.read_binary_design().tocsc()
            cov_freq = np.array(X_bin.mean(0)).ravel()
            bin_convariate_name = bin_convariate_name[cov_freq > self.min_freq]

        covariate_name = np.concatenate((
            bin_convariate_name, cont_covariate_name
        ))

        if self.include_interaction:
            interaction_name = np.array([
                "dabigatran interaction with " + name
                for name in covariate_name
            ])
            if self.exclude_main_effect:
                covariate_name = interaction_name
            else:
                covariate_name = np.concatenate((covariate_name, interaction_name))

        if self.regress_on == 'linearly_adjusted_outcome':
            covariate_name = np.concatenate((
                ['treatment by dabigatran'], covariate_name
            ))
        elif self.regress_on == 'outcome':
            covariate_name = np.concatenate((
                ['treatment by dabigatran'],
                ['propensity_strata_' + str(i + 2)
                 for i in range(self.n_strata - 1)],
                covariate_name
            ))
        else:
            NotImplementedError()

        return covariate_name

    def search_covariate_indices(
            self, keywords, exclude_words=None, covariate_name=None):
        if covariate_name is None:
            covariate_name = self.read_covariate_name()

        keywords = np.atleast_1d(keywords)
        keywords = np.array([word.lower() for word in keywords])
        if exclude_words is not None:
            exclude_words = np.atleast_1d(exclude_words)
            exclude_words = np.array([word.lower() for word in exclude_words])

        covariate_index = np.where(np.array([
            self.has_keyword(name.lower(), keywords, exclude_words)
            for name in covariate_name
        ]))[0]

        if covariate_index.size == 1:
            covariate_index = covariate_index.item()

        return covariate_index

    @staticmethod
    def has_keyword(string, keywords, exclude_words=None):
        """ Check that all the keywords matche and that none of
        the exclusion words match. """
        keyword_found = np.all([word in string for word in keywords])
        if exclude_words is None:
            exclude_word_found = False
        else:
            exclude_word_found = np.any([
                word in string for word in exclude_words
            ])
        return keyword_found and (not exclude_word_found)

    def save_gibbs_output(
            self, mcmc_output, mvnorm_method='', suffix=None):
        if mcmc_output['global_scale_update'] is None:
            gscale = mcmc_output['_markov_chain_state']['global_scale']
        else:
            gscale = None
        bridge_exponent = mcmc_output['bridge_exponent']
        file = self.get_gibbs_output_filename(
            mvnorm_method, gscale, bridge_exponent, suffix
        )
        # Set the non-picklable items to None.
        non_picklable = {
            key: mcmc_output[key]
            for key in ['_random_gen_state', '_reg_coef_sampler_state']
        }
        for key in non_picklable.keys():
            mcmc_output[key] = None

        with open(self.folder + file, 'wb') as f:
            print("Saving the output as {:s}...".format(file))
            pkl.dump(mcmc_output, f)

        # Restore the non-picklable items.
        for key in non_picklable.keys():
            mcmc_output[key] = non_picklable[key]
        return

    def get_gibbs_output_filename(
            self, mvnorm_method, gscale=None, bridge_exponent=.5, suffix=None):
        file = self.get_model_name(gscale, bridge_exponent)
        file += "_samples"
        if mvnorm_method == 'pcg' or mvnorm_method == 'cg':
            file = 'pcg_' + file
        if suffix is not None:
            file += '_' + suffix
        file += ".pkl"
        return file

    def get_model_name(self, gscale=None, bridge_exponent=.5):
        file = "bayes_bridge_on_"
        if bridge_exponent != .5:
            file = '{:d}th_exponent_'.format(int(1 / bridge_exponent)) + file
        if 'simulated' not in self.regress_on.split('_'):
            file += "OHDSI_"
        file += self.regress_on
        if self.min_freq > 0:
            file += "_covariate_filtered"
        file += "_model"
        if self.exclude_main_effect:
            file += '_without_main_effect'
        if self.include_interaction:
            file += '_with_interaction'
        if self.center_treatment_only:
            file += '_treatment_only_centered'
        if gscale is not None:
            file += '_{:d}th_global_scale'.format(int(1 / gscale))
        return file

    def load_gibbs_output(
            self, mvnorm_method='', gscale=None, bridge_exp=.5, suffix=None):
        file = self.get_gibbs_output_filename(
            mvnorm_method, gscale, bridge_exp, suffix
        )
        with open(self.folder + file, 'rb') as f:
            print("Loading the file {:s}...".format(file))
            mcmc_output = pkl.load(f)

        samples = mcmc_output['samples']
        if 'tau' in samples:
            samples['global_scale'] = samples['tau']
        if 'lambda' in samples:
            samples['local_scale'] = samples['lambda']

        if 'n_pcg_iter' in mcmc_output:
            mcmc_output['reg_coef_sampling_info'] \
                = {'n_cg_iter': mcmc_output['n_pcg_iter']}

        return mcmc_output

    def save_pcg_simulation(self, cg_output, precond_by, include_intercept=True):
        file = self.get_pcg_simulation_filename(precond_by, include_intercept)
        with open(self.folder + file, 'wb') as file:
            pkl.dump(cg_output, file)
        return

    def get_pcg_simulation_filename(
            self, precond_by, include_intercept=True):
        if self.use_gi_outcome:
            file = self.get_model_name()
        else:
            # For backward compatibility
            file = "bayes_bridge_on_OHDSI_" + self.regress_on
        file += "_" + precond_by.replace('+', '_') + "_precond_cg"
        if not include_intercept:
            file += "_wo_intercept"
        file += "_outputs.pkl"
        return file

    def load_pcg_simulation(self, precond_by, include_intercept=True):
        file = self.get_pcg_simulation_filename(precond_by, include_intercept)
        with open(self.folder + file, 'rb') as file:
            cg_output = pkl.load(file)
        return cg_output

    def save_precond_eigvals(self, eigvals, precond_by):
        filename = self.get_precond_eigval_filename(precond_by)
        np.save(self.folder + filename, eigvals)

    def get_precond_eigval_filename(self, precond_by):
        file = '_'.join([
            "bayes_bridge_on_OHDSI", self.regress_on,
            precond_by.replace('+', '_'), "precond_eigvalues.npy",
        ])
        return file

    def load_precond_eigvals(self, precond_by):
        filename = self.get_precond_eigval_filename(precond_by)
        return np.load(self.folder + filename)