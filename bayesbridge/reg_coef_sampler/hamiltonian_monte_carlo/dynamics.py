import numpy as np
import math


"""
Defines a (numerical) Hamiltonian dynamics based on a Gaussian momentum and the 
velocity Verlet integrator. The code is written so that other integrators & 
momentum distributions can also be employed straightwardly.
"""

class HamiltonianDynamics():

    def __init__(self, mass=None):
        """
        Parameters
        ----------
        mass(p, power) : callable
            Returns a vector obtained by multiplying the vector p with matrix
            M ** power for power == -1 or power == 1/2. The matrix L corresponding
            to M ** 1/2 only needs to satisfy L L' = M. Passing M = None defaults
            to a dynamics with the identity mass matrix.
        """
        self.integrator = velocity_verlet
        self.momentum = GaussianMomentum(mass)

    def integrate(self, f, dt, q, p, grad):
        q, p, logp, grad \
            = velocity_verlet(f, self.momentum.get_grad, dt, q, p, grad)
        return q, p, logp, grad

    def draw_momentum(self, n_param):
        return self.momentum.draw_random(n_param)

    def compute_hamiltonian(self, logp, p):
        potential = - logp
        kinetic = - self.momentum.get_logp(p)
        return potential + kinetic


def velocity_verlet(
        get_position_logp_and_grad, get_momentum_grad, dt, q, p, position_grad
    ):
    p = p + 0.5 * dt * position_grad
    q = q - dt * get_momentum_grad(p)
    position_logp, position_grad = get_position_logp_and_grad(q)
    if math.isfinite(position_logp):
        p += 0.5 * dt * position_grad
    return q, p, position_logp, position_grad


class GaussianMomentum():

    def __init__(self, mass=None):
        if mass is None:
            mass = lambda p, power: p
        self.mass = mass

    def draw_random(self, n_param):
        p = self.mass(np.random.randn(n_param), 1/2)
        return p

    def get_grad(self, p):
        return - self.mass(p, -1)

    def get_logp(self, p):
        return - 0.5 * np.dot(p, self.mass(p, -1))
