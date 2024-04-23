import copy
from math import exp, floor, log, log10, isclose, sqrt
import numpy as np

import argparse

# coverage probability
P_0 = 0.95

RJ = 37

n_digs = 3


def _mean(pdf: list, h: float) -> float:
    """
    calculate a mean value for given pdf
    :param pdf: the pdf values
    :param h: integration step
    :return: mean value
    """

    f = copy.deepcopy(pdf)
    for i in range(len(f)):
        f[i] *= (i * h)

    return np.trapz(f, dx=h)


def _stdev(pdf: list, mean: float, h: float) -> float:
    """
    calculate a standard deviation for given pdf
    :param pdf: the pdf values
    :param mean: a pre-calculated mean value
    :param h: integration step
    :return: stdev value
    """

    f = copy.deepcopy(pdf)
    for i in range(len(f)):
        f[i] *= (i * h - mean) ** 2

    return sqrt(np.trapz(f, dx=h))


def _next_I(traps: list, integral: float, i_start: int, i_end: int) -> tuple:
    """
    an auxiliary function to calculate coverage interval
    """

    integral -= traps[i_start]
    i_start += 1
    while (integral < P_0) and (i_end < len(traps) - 1):
        i_end += 1
        integral += traps[i_end]

    new_integral = integral if integral >= P_0 else None
    return new_integral, i_start, i_end


def _K(pdf: list, h: float) -> tuple:

    n_traps = len(pdf) - 1
    traps = [0] * n_traps
    for i in range(n_traps):
        traps[i] = 0.5 * (pdf[i] + pdf[i + 1]) * h

    traps.reverse()
    I = 0.
    i_start = 0
    i_end = 0
    for i_end in range(1, n_traps):
        I += traps[i_end]
        if I > P_0:
            break

    ci_lengths = {(i_start, i_end): i_end - i_start}
    while True:
        I, i_start, i_end = _next_I(traps, I, i_start, i_end)
        if I is None:
            break
        ci_lengths[(i_start, i_end)] = i_end - i_start

    i_2, i_1 = min(ci_lengths, key=ci_lengths.get)
    start = 1 - i_1 * h
    end = 1. - i_2 * h
    return start, end


def _symmetric_coverage_interval(pdf: list, x: float, u: float, k: float, h: float) -> None:
    """
    calculate a symmetric coverage interval for given pdf
    :param pdf: the pdf values
    :param x: a measured value
    :param u: a measurement uncertainty
    :param k: a coverage factor (1.96 or 2)
    :param h: integration step
    """

    start_init = x - k * u
    start = max(start_init, 0.)
    end_init = x + k * u
    end = min(end_init, 1.)

    cut = "" if (isclose(start, start_init) and isclose(end, end_init)) else "(cut)"

    i_start = round(start / h)
    i_end = round(end / h)
    f = pdf[i_start : i_end]
    prob = round(np.trapz(f, dx=h), 3)

    p0 = "{:.1f}%".format(100. * P_0)

    validity = f"  - Invalid! P < {p0}" if (prob < P_0 - 1.e-3) else ""

    # to str
    start = ("{:." + str(n_digs) + "f}").format(start)
    end = ("{:." + str(n_digs) + "f}").format(end)
    prob = "{:.1f}%".format(100. * prob)

    interval = f"({start}, {end}), P = {prob}"
    out = f"symmetric {cut} coverage interval ".ljust(RJ) + interval + validity
    print(out)


def _stats(pdf: list, h: float) -> tuple:

    m = _mean(pdf, h)
    sigma = _stdev(pdf, m, h)
    k_start, k_end = _K(pdf, h)

    return m, sigma, k_start, k_end


def get_pdf(h: float, x: float, u: float, c_0: float, w: float, prior_tsp: bool) -> list:
    """

    :param h: integration step
    :param x: a measured value
    :param u: a measurement uncertainty
    :param c_0: the "minimal expected" value
    :param w: a weight assigned to [c_0, 1] interval
    :param prior_tsp: use asymmetric TSP distribution as a prior pdf if True; otherwise use uniform prior pdf
    :return:
    """

    n_vals = round(1. / h) + 1
    p = log(1. - w) / log(c_0)

    res = [0] * n_vals
    for i in range(n_vals):

        v = i * h

        if prior_tsp:
            res[i] = exp(-0.5 * ((v - x) / u) ** 2) * p * v ** (p - 1.)
        else:
            res[i] = 0 if v < c_0 else exp(-0.5 * ((v - x) / u) ** 2)

    res = list(np.array(res) * (1. / np.trapz(res, dx=h)))  # norming
    return res


def main(x: float, u: float, c_0: float, w: float, k: float):

    n_digs_u = -floor(log10(u))
    h = 10 ** (-n_digs_u - 3)

    global n_digs
    n_digs = n_digs_u + 1

    print()
    print("input data")
    print("---------------------------------------------")
    print("x".ljust(RJ) + (" {:." + str(n_digs) + "f}").format(x))
    print("u".ljust(RJ) + (" {:." + str(n_digs) + "f}").format(u))
    print("c0".ljust(RJ) + (" {:." + str(n_digs) + "f}").format(c_0))
    print("w".ljust(RJ) + (" {:." + str(2) + "f}").format(w))
    print()

    alpha = (1. - c_0) / u
    beta = (1. - x) / u

    prior_tsp = (alpha <= 5) or (beta > alpha - 3)

    pdf = get_pdf(h, x, u, c_0, w, prior_tsp)
    assert isclose(h * sum(pdf), 1., abs_tol=0.01)

    stats = _stats(pdf, h)

    max_val = max(pdf)
    mode = h * pdf.index(max_val)

    print("posterior pdf")
    print("---------------------------------------------")


    ln = "mean".ljust(RJ) + (" {:." + str(n_digs) + "f}").format(stats[0])
    print(ln)
    ln = "mode".ljust(RJ) + (" {:." + str(n_digs) + "f}").format(mode)
    print(ln)
    ln = "stdev".ljust(RJ) + (" {:." + str(n_digs) + "f}").format(stats[1])
    print(ln)


    ci = "("
    ci += ("{:." + str(n_digs) + "f}").format(stats[2])
    ci += ", "
    ci += ("{:." + str(n_digs) + "f}").format(stats[3])
    ci += f"),  P = "
    ci += ("{:.1f}%").format(100 * P_0)
    print("shortest coverage interval ".ljust(RJ) + ci)
    _symmetric_coverage_interval(pdf, x, u, k, h)

    print()


if __name__ == "__main__":

    # TODO: checks (x in [0, 1], u > 0, u < ?, w in [0.5, 0.99], ...)

    parser = argparse.ArgumentParser()
    parser.add_argument("--x", required=True, type=float)
    parser.add_argument("--u", required=True, type=float)
    parser.add_argument("--c0", required=True, type=float)
    parser.add_argument("--w", required=False, type=float, default=0.75)
    parser.add_argument("--k", required=False, type=float, default=2.)
    # P0

    args = parser.parse_args()

    main(args.x, args.u, args.c0, args.w)
