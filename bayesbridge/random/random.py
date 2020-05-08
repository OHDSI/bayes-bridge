import numpy as np
from .polya_gamma import PolyaGammaDist
from .tilted_stable import ExpTiltedStableDist

class BasicRandom():
    """
    Generators of random variables from the basic distributions used in
    Bayesian sparse regression.
    """

    def __init__(self, seed=None):
        self.np_random = np.random
        self.pg = PolyaGammaDist()
        self.ts = ExpTiltedStableDist()
        self.set_seed(seed)

    def set_seed(self, seed):
        self.np_random.seed(seed)
        pg_seed = np.random.randint(1, 1 + np.iinfo(np.int32).max)
        ts_seed = np.random.randint(1, 1 + np.iinfo(np.int32).max)
        self.pg.set_seed(pg_seed)
        self.ts.set_seed(ts_seed)

    def get_state(self):
        rand_gen_state = {
            'numpy' : self.np_random.get_state(),
            'tilted_stable' : self.ts.get_state(),
            'polya_gamma' : self.pg.get_state()
        }
        return rand_gen_state

    def set_state(self, rand_gen_state):
        self.np_random.set_state(rand_gen_state['numpy'])
        self.ts.set_state(rand_gen_state['tilted_stable'])
        self.pg.set_state(rand_gen_state['polya_gamma'])

    def polya_gamma(self, shape, tilt):
        return self.pg.rand_polyagamma(shape, tilt)

    def tilted_stable(self, char_exponent, tilt):
        return self.ts.sample(char_exponent, tilt)
