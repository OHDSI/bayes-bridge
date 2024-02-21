# source /Users/gracexd/.virtualenvs/bayesbridge_test_pypi/bin/activate

import numpy as np
import time
from typing import Optional
from helper import log_prob_c_pos, log_prob_c_neg
import logging



def compute_inflection_point(c: float) -> Optional[float]:
    sqrt3 = np.sqrt(3)
    cutoff = 3 * sqrt3
    if 0 <= c <= cutoff:
        return None
    elif 0 > c >= -cutoff:
        c_reciprocal = 1 / c
        delta = np.sqrt(c_reciprocal ** 2 - 1 / 27)
        r = (-c_reciprocal + delta) ** (1 / 3) + (- c_reciprocal - delta) ** (1 / 3)
    else:
        theta = np.arccos(cutoff / c)
        t1 = np.cos(theta / 3) / sqrt3
        t2 = np.sin(theta / 3)
        if c > cutoff:
            r = np.array([t1 + t2, t1 - t2])
        else:
            r = np.array([t1 + t2])
    return 1 / r


def skewed_shrinkage_rejection_sampler(a: float, c: float, q: float, k1: int, k2: int):
    accepted = False
    i = 0
    while not accepted:
        i += 1
        if c >= 0:
            logging.info("Sampling from c positive")
            res = sample_c_positive(a, c, q, k1, k2)
            bound_log_prob = res[1]
            rv = res[0]
            target_log_prob = log_prob_c_pos(rv, a, c)
        else:
            logging.info("Sampling from c negative")
            res = sample_c_negative(a, c, q, k1, k2)
            bound_log_prob = res[1]
            rv = res[0]
            target_log_prob = log_prob_c_neg(rv, a, c)
        accept_prob = np.exp(target_log_prob - bound_log_prob)
        accepted = accept_prob > np.random.uniform(0, 1)
    return rv, i


def skewed_shrinkage_rejection_sampler_sim(a: float, c: float, nsim: int, q: float, k1: int, k2: int):
    rv = np.zeros(nsim)
    ar = np.zeros(nsim)
    for ind in range(nsim):
        rv[ind], ar[ind] = skewed_shrinkage_rejection_sampler(a, c, q, k1, k2)
    return rv, 1 / np.mean(ar)


def x_plus_reciprocal(x):
    return x + 1 / x


def h_derivative(a_sq, c, x_transformed):
    return -2 * a_sq * (x_transformed - c) * x_plus_reciprocal(x_transformed)


def mass_exp_approx(d_concave, den, x):
    coef = 1 - np.exp(- d_concave * x)
    integral = den / d_concave * coef
    return coef, integral


def mass_ziggurat(logq, k0, k1, x):
    log_den_const = logq * abs(np.arange(-k0, k1 + 1))
    den_const = np.exp(log_den_const)
    int_const = np.concatenate([den_const[1:(k0 + 1)] * (x[1:(k0 + 1)] - x[:k0]),
                                den_const[k0:(k0 + k1)] * (x[(k0 + 1):(k0 + k1 + 1)] - x[k0:(k0 + k1)])])
    return log_den_const, den_const, int_const


def mass_concave_left_to_ziggurat(log_den_x, den_x, x_trans, x, a):
    d_concave = 2 * a * np.sqrt(abs(log_den_x)) * x_plus_reciprocal(x_trans)
    coef_concave, int_concave = mass_exp_approx(d_concave, den_x, x)
    return d_concave, coef_concave, int_concave


def mass_leftmost_concave(a_sq, c, r, r_transformed):
    tmp = c - r_transformed
    d_concave_left = h_derivative(a_sq, c, r_transformed)
    log_den_concave_left = - a_sq * tmp ** 2
    den_concave_left = np.exp(log_den_concave_left)
    coef_concave_left, int_concave_left = mass_exp_approx(d_concave_left, den_concave_left, r)
    return d_concave_left, log_den_concave_left, coef_concave_left, int_concave_left


def mass_log_convex_c_pos(m1, m2, k, a, c):
    m_convex = np.linspace(m1, m2, num=k + 1)
    log_den_convex = log_prob_c_pos(x=m_convex, a=a, c=c)
    den_convex = np.exp(log_den_convex)
    d_convex = (log_den_convex[1:(k + 1)] - log_den_convex[:k]) / (m_convex[1:(k + 1)] - m_convex[:k])
    coef, int_convex = mass_exp_approx(d_convex, den_convex[1:(k + 1)], m_convex[1:(k + 1)] - m_convex[:k])
    return m_convex, log_den_convex, d_convex, coef, int_convex


def mass_log_convex_c_neg(m1, m2, k, a, c):
    m_convex = np.linspace(m1, m2, num=k + 1)
    log_den_convex = log_prob_c_neg(x=m_convex, a=a, c=c)
    den_convex = np.exp(log_den_convex)
    d_convex_opp = - (log_den_convex[1:(k + 1)] - log_den_convex[:k]) / (m_convex[1:(k + 1)] - m_convex[:k])
    coef, int_convex = mass_exp_approx(d_convex_opp, den_convex[:k], x=m_convex[1:(k + 1)] - m_convex[:k])
    return m_convex, log_den_convex, den_convex, d_convex_opp, coef, int_convex


def compute_rv_and_log_prob(uniform_rlim, x, log_den, derivative):
    rv = np.random.uniform(low=0, high=uniform_rlim, size=1)
    rv = x + np.log(1 - rv) / derivative
    log_prob = log_den + derivative * (rv - x)
    return rv, log_prob


def right_log_concave(a_sq,c, x_transformed, den_x):
    d_concave_opp = - h_derivative(a_sq, c, x_transformed)
    int_concave = den_x / d_concave_opp
    return d_concave_opp, int_concave


def sample_c_positive(a: float, c: float, q: float, k1: int, k2: int):
    logq = np.log(q)
    sqrt_neg_logq = np.sqrt(-logq)
    a_sq = a ** 2

    # k0 corresponds to the point where f(0) = exp(-a^2 c^2) == q^k0
    # if q^k1 <= q^k0 <=> k1 >= k0, then there's no part on the left to the ziggurat region,

    # Compute how many constant pieces we can use on the left side of the mode.
    # If f(0) >= q^k1, we can use piecewise constant proposal to upper bound the
    # left side of the mode with guaranteed acceptance rate; otherwise, we need
    # to compute where the target density is concave/convex and use exponential proposal.

    k0 = - a_sq * c ** 2 / logq
    if k0 <= k1:
        l_computed = False
        k0 = int(k0 // 1 + 1)
    else:
        l_computed = True
        k0 = k1

    x_transformed = c + sqrt_neg_logq / a * np.concatenate([-np.sqrt(np.arange(k0, 0, -1)),
                                                            np.sqrt(np.arange(0, k1 + 1))])
    if not l_computed:
        # which means that 1) there are no regions on the left of the ziggurat region,
        # 2) the ziggurat region starts from x = 0 <=> x_transformed = sqrt(e^2x -1) = 0
        x_transformed[0] = 0
    x = np.log(1 + x_transformed ** 2) / 2

    # proposed ziggurat region density
    # can be optimized by shortening the length, -k0 and k1 are not used in calculation
    log_den_const, den_const, int_const = mass_ziggurat(logq, k0, k1, x)

    # right tail
    # x2_t:= x_transformed[k0+k1]
    # proposing tangent line
    # d_concave_opp = - h_derivative(a_sq, c, x_transformed[k0 + k1])
    # int_concave_right = den_const[k0 + k1] / d_concave_opp
    d_concave_opp, int_concave_right = right_log_concave(a_sq,c, x_transformed[k0 + k1], den_const[k0 + k1])

    # left tail if l_computed == True
    int_concave_left = 0
    int_convex = np.zeros(k2)
    int_concave_mid = 0

    # if there are log-concave or log-convex region on the left,
    # then we
    if l_computed:
        r_transformed = compute_inflection_point(c)
        r = np.log(1 + r_transformed ** 2) / 2

        if (r_transformed is None) or (r[0] >= x[0]):
            d_concave_left, coef_concave_left, int_concave_left = \
                mass_concave_left_to_ziggurat(log_den_x=log_den_const[0], den_x=den_const[0],
                                              x_trans=x_transformed[0],x=x[0], a=a)

        else:
            # the leftmost log-concave region
            d_concave_left, log_den_concave_left, coef_concave_left, int_concave_left = \
                mass_leftmost_concave(a_sq=a_sq, c=c, r=r[0], r_transformed=r_transformed[0])
            # the middle log-convex region - split into k2 parts
            # print(k2 + 1)
            m_convex, log_den_convex, d_convex, coef, int_convex = \
                mass_log_convex_c_pos(m1=r[0],m2=np.min(r[1], x[0]),k=k2, a=a, c=c)

            if r[1] < x[0]:
                d_concave_mid, coef_concave_mid, int_concave_mid = mass_concave_left_to_ziggurat(
                    log_den_x=log_den_const[0],
                    den_x=den_const[0],
                    x_trans=x_transformed[0],
                    x=x[0] - r[0],
                    a=a)

    ints = np.concatenate([np.array([int_concave_left]),
                           int_convex,
                           np.array([int_concave_mid]),
                           int_const,
                           np.array([int_concave_right])], axis=None)
    probs = ints / np.sum(ints)

    # print(ints)
    # print(probs)

    # sample from the proposed distribution

    piece = np.random.choice(len(probs), size=1, p=probs)

    if piece == 0:
        if (r_transformed is None) or (r[0] >= x[0]):
            logging.info("Sampling from piece 0")
            rv, log_prob = compute_rv_and_log_prob(coef_concave_left, x[0], log_den_const[0], d_concave_left)
        else:
            rv, log_prob = compute_rv_and_log_prob(coef_concave_left, r[0], log_den_concave_left, d_concave_left)
    elif piece <= k2:
        logging.info("Sampling from piece 0-k2")
        piece = piece - 1
        rv, log_prob = compute_rv_and_log_prob(np.squeeze(coef[piece]), m_convex[piece + 1],
                                               log_den_convex[piece + 1], d_convex[piece])
    elif piece <= (1 + k2):
        logging.info("Sampling from piece k2")
        rv, log_prob = compute_rv_and_log_prob(coef_concave_mid, x[0], log_den_const[0], d_concave_mid)
    elif piece <= (1 + k2 + k0 + k1):
        logging.info("Sampling from piece k2 - 1 + k2 + k0 + k1")
        piece = piece - 2 - k2
        rv = np.random.uniform(low=x[piece], high=x[piece + 1], size=1)
        if piece <= k0 - 1:
            log_prob = log_den_const[piece + 1]
        else:
            log_prob = log_den_const[piece]
    else:
        logging.info("Sampling from right")
        rv = x[k0 + k1] + np.random.exponential(scale=1 / d_concave_opp, size=1)
        log_prob = log_den_const[k0 + k1] - d_concave_opp * (rv - x[k0 + k1])
    return rv, log_prob


def sample_c_negative(a: float, c: float, q: float, k1: int, k2: int):
    logq = np.log(q)
    a_sq = a ** 2

    x_transformed = np.sqrt(c ** 2 - logq / a_sq * np.arange(k1 + 1)) + c

    # if underflow occurs, you can try using talyor expansion to approximate these points:
    # if x_transformed[k1+2] == 0:
    #     x_transformed = - np.arange(k1+1) * neg_log_q / 2 / a_sq / c

    x = np.log(1 + x_transformed ** 2) / 2

    # compute the inflection points where the target density changes from
    # convex to concave
    r_transformed = compute_inflection_point(c)

    # if r_transformed > m_transformed[k1+2], we keep it;
    # otherwise, it is not used so we do not need to compute r.
    l_computed = False
    if r_transformed > x_transformed[k1]:
        l_computed = True
        r = np.log(1 + r_transformed ** 2) / 2

    log_den = logq * np.arange(k1 + 1)
    den = np.exp(log_den)
    int_const = den[:k1] * (x[1:(k1 + 1)] - x[:k1])

    if not l_computed:
        # log-convex part
        int_convex = np.zeros(k2)
        # log-concave part
        d_concave_opp, int_concave = right_log_concave(a_sq, c, x_transformed[k1], den[k1])
    else:
        # log-convex part
        m_convex, log_den_convex, den_convex, \
            d_convex_opp, coef, int_convex = mass_log_convex_c_neg(m1=x[k1], m2=r, k=k2, a=a, c=c)
        # log-concave part
        # d_concave_opp = 2 * a_sq * (1 + r_transformed ** 2) * (1 - c / r_transformed)
        # int_concave = den_convex[k2] / d_concave_opp
        d_concave_opp, int_concave = right_log_concave(a_sq, c, r_transformed, den_convex[k2])

    ints = np.concatenate([int_const, int_convex, np.array([int_concave])], axis=None)
    probs = ints / sum(ints)

    piece = np.random.choice(len(probs), size=1, p=probs)

    if piece <= (k1 - 1):
        logging.info("Sampling from 0 - k1-1")
        rv = np.random.uniform(x[piece], x[piece + 1])
        log_prob = log_den[piece]
    elif piece <= (k1 + k2 - 1):
        logging.info("Sampling from k1-1 - k1+k2-1")
        piece = piece - k1
        # print(coef[piece].shape)
        rv, log_prob = compute_rv_and_log_prob(np.squeeze(coef[piece]), m_convex[piece],
                                               log_den_convex[piece], -d_convex_opp[piece])
    else:
        logging.info("Sampling from right")
        if not l_computed:
            rv = x[k1] + np.random.exponential(scale=1 / d_concave_opp)
            log_prob = log_den[k1] - d_concave_opp * (rv - x[k1])
        else:
            rv = r + np.random.exponential(scale=1 / d_concave_opp)
            log_prob = log_den_convex[k2] + d_concave_opp * (rv - r)
    return rv, log_prob


# def sample_from_ziggurat():
#     return

def main():
    logging.basicConfig(level=logging.WARNING,  # Change level to INFO
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Adding a file handler
    file_handler = logging.FileHandler('app.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logging.getLogger().addHandler(file_handler)

    print("c positive")
    a = 1
    c = 0
    Nsim = 10 ** 5
    q = 0.5
    k1 = 16
    k2 = 5
    start_time = time.time()
    sample, acceptance_rate = skewed_shrinkage_rejection_sampler_sim(a, c, Nsim, q, k1, k2)
    print(time.time() - start_time)
    print(acceptance_rate)

    # print("c positive")
    # a = 0.4268421
    # c = 8.631319
    # Nsim = 10 ** 5
    # q = 0.5
    # k1 = 16
    # k2 = 5
    # start_time = time.time()
    # sample, acceptance_rate = skewed_shrinkage_rejection_sampler_sim(a, c, Nsim, q, k1, k2)
    # print(time.time() - start_time)
    # print(acceptance_rate)
    # # print(sample)

    # print("c negative")
    # a = 1000
    # c = -100
    # Nsim = 100
    # q = 0.5
    # k1 = 16
    # k2 = 5
    # start_time = time.time()
    # sample, acceptance_rate = skewed_shrinkage_rejection_sampler_sim(a, c, Nsim, q, k1, k2)
    # print(time.time() - start_time)
    # print(acceptance_rate)


if __name__ == "__main__":
    main()
