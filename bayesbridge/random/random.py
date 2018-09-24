import numpy as np
from pypolyagamma import PyPolyaGamma
from .tilted_stable_dist.rand_exp_tilted_stable import ExpTiltedStableDist

class BasicRandom():
    """
    Generators of random variables from the basic distributions used in
    Bayesian sparse regression.
    """

    def __init__(self, seed=None):
        self.np_random = np.random
        self.pg = None
        self.ts = None
        self.set_seed(seed)

    def set_seed(self, seed):
        self.np_random.seed(seed)
        pg_seed = np.random.randint(1, 1 + np.iinfo(np.uint32).max)
        ts_seed = np.random.randint(1, 1 + np.iinfo(np.uint32).max)
        self.pg = PyPolyaGamma(seed=pg_seed)
        self.ts = ExpTiltedStableDist(seed=ts_seed)

    def get_state(self):
        rand_gen_state = {
            'numpy' : self.np_random.get_state(),
            'tilted_stable' : self.ts.get_state(),
            'pypolyagamma' : self.pg
                # Don't know how to access the internal state, so just save
                # the object itself.
        }
        return rand_gen_state

    def set_state(self, rand_gen_state):
        self.np_random.set_state(rand_gen_state['numpy'])
        self.ts.set_state(rand_gen_state['tilted_stable'])
        self.pg = rand_gen_state['pypolyagamma']

    def polya_gamma(self, shape, tilt, size):
        omega = np.zeros(size)
        self.pg.pgdrawv(shape, tilt, omega)
        return omega

    def tilted_stable(self, char_exponent, tilt):
        return self.ts.rv(char_exponent, tilt)
