import numpy as np
import math
import time
from .stepsize_adapter import HamiltonianBasedStepsizeAdapter, initialize_stepsize
from .dynamics import HamiltonianDynamics
from .util import warn_message_only


class NoUTurnSampler():

    def __init__(self, f, mass=None, warning_requested=True):
        """
        Parameters
        ----------
        f: callable
            Return the log probability and gradient evaluated at q.
        mass: None, numpy 1d array, or callable
        """
        self.f = f
        self.dynamics = HamiltonianDynamics(mass)
        self.warning_requested = warning_requested

    def generate_samples(
            self, q0, n_burnin, n_sample, dt_range=None, seed=None, n_update=0,
            adapt_stepsize=False, target_accept_prob=.9, final_adaptsize=.05):
        """
        Implements the No-U-Turn Sampler (NUTS) of Hoffman and Gelman (2011).

        Parameters:
        -----------
        dt_range: None, float, or ndarray of length 2
        adapt_stepsize: bool
            If True, the max stepsize will be adjusted to to achieve the target
            acceptance rate. Forced to be True if dt_range is None.
        """

        if seed is not None:
            np.random.seed(seed)

        q = q0
        logp, grad = self.f(q)

        if np.isscalar(dt_range):
            dt_range = np.array(2 * [dt_range])

        elif dt_range is None:
            p = self.dynamics.draw_momentum(len(q))
            logp_joint0 = - self.dynamics.compute_hamiltonian(logp, p)
            dt = initialize_stepsize(
                lambda dt: self.compute_onestep_accept_prob(dt, q, p, grad, logp_joint0)
            )
            dt_range = dt * np.array([.8, 1.0])
            adapt_stepsize = True

        max_stepsize_adapter = HamiltonianBasedStepsizeAdapter(
            init_stepsize=1., target_accept_prob=target_accept_prob,
            reference_iteration=n_burnin, adaptsize_at_reference=final_adaptsize
        )

        if n_update > 0:
            n_per_update = math.ceil((n_burnin + n_sample) / n_update)
        else:
            n_per_update = float('inf')
        samples = np.zeros((len(q), n_sample + n_burnin))
        logp_samples = np.zeros(n_sample + n_burnin)
        accept_prob = np.zeros(n_sample + n_burnin)
        max_dt = np.zeros(n_burnin)

        tic = time.time()
        use_averaged_stepsize = False
        for i in range(n_sample + n_burnin):
            dt_multiplier \
                = max_stepsize_adapter.get_current_stepsize(use_averaged_stepsize)
            dt = np.random.uniform(dt_range[0], dt_range[1])
            dt *= dt_multiplier
            q, info = self.generate_next_state(dt, q, logp, grad)
            logp, grad = info['logp'], info['grad']
            if i < n_burnin and adapt_stepsize:
                max_dt[i] = dt_range[1] * dt_multiplier
                max_stepsize_adapter.adapt_stepsize(info['ave_hamiltonian_error'])
            elif i == n_burnin - 1:
                use_averaged_stepsize = True
            samples[:, i] = q
            logp_samples[i] = logp
            if (i + 1) % n_per_update == 0:
                print('{:d} iterations have been completed.'.format(i + 1))

        toc = time.time()
        time_elapsed = toc - tic

        info = {
            'logp_samples': logp_samples,
            'accept_prob_samples': accept_prob,
            'sampling_time': time_elapsed
        }
        if adapt_stepsize:
            info['max_stepsize'] = max_dt

        return samples, info


    def compute_onestep_accept_prob(self, dt, q0, p0, grad0, logp_joint0):
        _, p, logp, _ = self.dynamics.integrate(self.f, dt, q0, p0, grad0)
        logp_joint = - self.dynamics.compute_hamiltonian(logp, p)
        accept_prob = np.exp(logp_joint - logp_joint0)
        return accept_prob

    def generate_next_state(self, dt, q, logp=None, grad=None, p=None,
                            max_height=10, hamiltonian_error_tol=100):

        n_grad_evals = 0
        if logp is None or grad is None:
            logp, grad = self.f(q)
            n_grad_evals += 1

        if p is None:
            p = self.dynamics.draw_momentum(len(q))

        logp_joint = - self.dynamics.compute_hamiltonian(logp, p)
        logp_joint_threshold = logp_joint - np.random.exponential()
            # Slicing variable in the log-scale.

        tree = _TrajectoryTree(
            self.dynamics, self.f, dt, q, p, logp, grad, logp_joint, logp_joint,
            logp_joint_threshold, hamiltonian_error_tol
        )
        directions = 2 * (np.random.rand(max_height) < 0.5) - 1
            # Pre-allocation of random directions is unnecessary, but makes the code easier to test.
        tree, final_height, last_doubling_rejected, maxed_before_u_turn \
            = self._grow_trajectory_till_u_turn(tree, directions)
        q, logp, grad = tree.sample
        n_grad_evals += tree.n_integration_step

        if self.warning_requested:
            self._issue_warnings(
                tree.instability_detected, maxed_before_u_turn, max_height
            )

        info = {
            'logp': logp,
            'grad': grad,
            'ave_accept_prob': tree.ave_accept_prob,
            'ave_hamiltonian_error': tree.ave_hamiltonian_error,
            'n_grad_evals': n_grad_evals,
            'tree_height': final_height,
            'u_turn_detected': tree.u_turn_detected,
            'instability_detected': tree.instability_detected,
            'last_doubling_rejected': last_doubling_rejected
        }

        return q, info

    def _issue_warnings(
            self, instability_detected, maxed_before_u_turn, max_height):

        if instability_detected:
            warn_message_only(
                "Numerical integration became unstable while simulating a "
                "NUTS trajectory."
            )
        if maxed_before_u_turn:
            warn_message_only(
                'The trajectory tree reached the max height of {:d} before '
                'meeting the U-turn condition.'.format(max_height)
            )
        return

    @staticmethod
    def _grow_trajectory_till_u_turn(tree, directions):

        height = 0 # Referred to as 'depth' in the original paper, but arguably the
                   # trajectory tree is built 'upward' on top of the existing ones.
        max_height = len(directions)
        trajectory_terminated = False
        while not trajectory_terminated:

            doubling_rejected \
                = tree.double_trajectory(height, directions[height])
                # No transition to the next half of trajectory takes place if the
                # termination criteria are met within the next half tree.

            height += 1
            trajectory_terminated \
                = tree.u_turn_detected or tree.instability_detected or (height >= max_height)
            maxed_before_u_turn \
                = height >= max_height and (not tree.u_turn_detected)

        return tree, height, doubling_rejected, maxed_before_u_turn


class _TrajectoryTree():
    """
    Collection of (a subset of) states along the simulated Hamiltonian dynamics
    trajcetory endowed with a binary tree structure.
    """

    def __init__(
            self, dynamics, f, dt, q, p, logp, grad, joint_logp,
            init_joint_logp, joint_logp_threshold, hamiltonian_error_tol=100.,
            u_turn_criterion='momentum'):

        self.dynamics = dynamics
        self.f = f
        self.dt = dt
        self.joint_logp_threshold = joint_logp_threshold
        self.front_state = (q, p, grad)
        self.rear_state = (q, p, grad)
        self.sample = (q, logp, grad)
        self.u_turn_detected = False
        self.min_hamiltonian = - joint_logp
        self.max_hamiltonian = - joint_logp
        self.hamiltonian_error_tol = hamiltonian_error_tol
        self.n_acceptable_state = int(joint_logp > joint_logp_threshold)
        self.n_integration_step = 0
        self.init_joint_logp = init_joint_logp
        self.height = 0
        self.ave_hamiltonian_error = abs(init_joint_logp - joint_logp)
        self.ave_accept_prob = min(1, math.exp(joint_logp - init_joint_logp))
        self.velocity_based_u_turn = (u_turn_criterion == 'velocity')

    @property
    def n_node(self):
        return 2 ** self.height

    @property
    def instability_detected(self):
        fluctuation_along_trajectory = self.max_hamiltonian - self.min_hamiltonian
        return fluctuation_along_trajectory > self.hamiltonian_error_tol

    def double_trajectory(self, height, direction):
        next_tree = self._build_next_tree(
            *self._get_states(direction), height, direction
        )
        no_transition_to_next_tree_attempted \
            = self._merge_next_tree(next_tree, direction, sampling_method='swap')
        return no_transition_to_next_tree_attempted

    def _build_next_tree(self, q, p, grad, height, direction):

        if height == 0:
            return self._build_next_singleton_tree(q, p, grad, direction)

        subtree = self._build_next_tree(q, p, grad, height - 1, direction)
        trajectory_terminated_within_subtree \
            = subtree.u_turn_detected or subtree.instability_detected
        if not trajectory_terminated_within_subtree:
            next_subtree = self._build_next_tree(
                *subtree._get_states(direction), height - 1, direction
            )
            subtree._merge_next_tree(next_subtree, direction, sampling_method='uniform')

        return subtree

    def _build_next_singleton_tree(self, q, p, grad, direction):
        q, p, logp, grad = \
            self.dynamics.integrate(self.f, direction * self.dt, q, p, grad)
        self.n_integration_step += 1
        if math.isinf(logp):
            joint_logp = - float('inf')
        else:
            joint_logp = - self.dynamics.compute_hamiltonian(logp, p)
        return self._clone_tree(q, p, logp, grad, joint_logp)

    def _clone_tree(self, q, p, logp, grad, joint_logp):
        """ Construct a tree with shared dynamics and acceptance criteria. """
        return _TrajectoryTree(
            self.dynamics, self.f, self.dt, q, p, logp, grad, joint_logp, self.init_joint_logp,
            self.joint_logp_threshold, self.hamiltonian_error_tol
        )

    def _merge_next_tree(self, next_tree, direction, sampling_method):

        # Trajectory termination flags from the next tree must be propagated up
        # the call stack, but other states of the tree is updated only if the
        # next tree is accessible from the current tree (i.e. the trajectory
        # did not get terminated within the next tree).

        self.u_turn_detected = self.u_turn_detected or next_tree.u_turn_detected
        self.min_hamiltonian = min(self.min_hamiltonian, next_tree.min_hamiltonian)
        self.max_hamiltonian = max(self.max_hamiltonian, next_tree.max_hamiltonian)
        trajectory_terminated_within_next_tree \
            = next_tree.u_turn_detected or next_tree.instability_detected

        if not trajectory_terminated_within_next_tree:
            self._update_sample(next_tree, sampling_method)
            self.n_acceptable_state += next_tree.n_acceptable_state
            self._set_states(*next_tree._get_states(direction), direction)
            self.u_turn_detected \
                = self.u_turn_detected or self._check_u_turn_at_front_and_rear_ends()
            weight = self.n_node / (self.n_node + next_tree.n_node)
            self.ave_hamiltonian_error \
                = weight * self.ave_hamiltonian_error + (1 - weight) * next_tree.ave_hamiltonian_error
            self.ave_accept_prob \
                = weight * self.ave_accept_prob + (1 - weight) * next_tree.ave_accept_prob
            self.height += 1

        return trajectory_terminated_within_next_tree

    def _update_sample(self, next_tree, method):
        """
        Parameters
        ----------
        method: {'uniform', 'swap'}
        """
        if method == 'uniform':
            n_total = self.n_acceptable_state + next_tree.n_acceptable_state
            sampling_weight_on_next_tree \
                = next_tree.n_acceptable_state / max(1, n_total)
        elif method == 'swap':
            sampling_weight_on_next_tree \
                = next_tree.n_acceptable_state / self.n_acceptable_state
        if np.random.uniform() < sampling_weight_on_next_tree:
            self.sample = next_tree.sample

    def _check_u_turn_at_front_and_rear_ends(self):
        q_front, p_front, _ = self._get_states(1)
        q_rear, p_rear, _ = self._get_states(-1)
        dq = q_front - q_rear
        if self.velocity_based_u_turn:
            v_front = self.dynamics.convert_to_velocity(p_front)
            v_rear = self.dynamics.convert_to_velocity(p_rear)
            u_turned = (np.dot(dq, v_front) < 0) or (np.dot(dq, v_rear) < 0)
        else:
            u_turned = (np.dot(dq, p_front) < 0) or (np.dot(dq, p_rear) < 0)
        return u_turned

    def _set_states(self, q, p, grad, direction):
        if direction > 0:
            self.front_state = (q, p, grad)
        else:
            self.rear_state = (q, p, grad)

    def _get_states(self, direction):
        if direction > 0:
            return self.front_state
        else:
            return self.rear_state
