from copy import copy
from math import exp, floor, log, log10, isclose, sqrt
import numpy as np

import argparse

# coverage probability
p_0 = 0.95
# default (initial) n of decimal digits
n_digs = 3

INDENT = 37  # for output formatting


def posterior_mean(pdf: list[float], h: float) -> float:
    """
    calculate a mean value for a given posterior pdf
    :param pdf: the list of pdf values
    :param h: integration step
    :return: the mean value
    """

    f = copy(pdf)
    for i in range(len(f)):
        f[i] *= (i * h)

    return np.trapz(f, dx=h)


def posterior_stdev(pdf: list[float], mean: float, h: float) -> float:
    """
    calculate a standard deviation (stdev) for a given posterior pdf
    :param pdf: the pdf values
    :param mean: a pre-calculated mean value
    :param h: integration step
    :return: the stdev value
    """

    f = copy(pdf)
    for i in range(len(f)):
        f[i] *= (i * h - mean) ** 2

    return sqrt(np.trapz(f, dx=h))


def _next_coverage_interval(traps: list[float], integral: float, i_start: int, i_end: int) -> tuple[float]:
    """
    An auxiliary function to calculate the next coverage interval
    :param traps: a list of trapezoids for the pdf (use a composite trapezoidal rule for the integration)
    :param integral: the value of the integral (sum of 'traps' from 'i_start' to 'i_end') on the previous step
    :param i_start: the previous starting index in 'traps' list
    :param i_end: the previous ending index in 'traps' list
    :return: a tuple containing the next values of the integral and starting and ending indices
    """

    # discard the 1st trapezoid
    integral -= traps[i_start]
    i_start += 1
    # then integrate until the value of the integral sum does not exceed p_0
    while (integral < p_0) and (i_end < len(traps) - 1):
        i_end += 1
        integral += traps[i_end]

    next_integral = integral if integral >= p_0 else None  # return None for the integral if cannot reach the p_0 value
    return next_integral, i_start, i_end


def shortest_coverage_interval(pdf: list[float], h: float) -> None:
    """
    Calculate and print the shortest coverage interval for the given pdf
    :param pdf: the list of pdf values
    :param h: integration step
    """

    n_traps = len(pdf) - 1
    traps = [0] * n_traps
    for i in range(n_traps):
        traps[i] = 0.5 * (pdf[i] + pdf[i + 1]) * h  # use composite trapezoid rule

    # the pdf is concentrated near x = 1, so let's mirror (start <-> end)
    traps.reverse()

    integral = 0.
    i_start = 0
    i_end = 0
    for i_end in range(1, n_traps):
        integral += traps[i_end]
        if integral > p_0:
            break

    # calculate all possible coverage intervals for given p_0
    ci_lengths = {(i_start, i_end): i_end - i_start}
    while True:
        integral, i_start, i_end = _next_coverage_interval(traps, integral, i_start, i_end)
        if integral is None:
            break
        ci_lengths[(i_start, i_end)] = i_end - i_start

    # choose the coverage interval having the shortest length
    i_2, i_1 = min(ci_lengths, key=ci_lengths.get)
    # borders of the shortest coverage interval
    ci_start = 1. - i_1 * h
    ci_end = 1. - i_2 * h

    fmt = "{:." + str(n_digs) + "f}"
    c1, c2 = fmt.format(ci_start), fmt.format(ci_end)
    p_0_pct = "{:.1f}%".format(100. * p_0)  # p_0 in %
    print("shortest coverage interval ".ljust(INDENT) + f"({c1}, {c2}), P = {p_0_pct}")


def print_symmetric_coverage_interval(pdf: list[float], x: float, u: float, k: float, h: float) -> None:
    """
    Calculate a symmetric coverage interval for given pdf and print it
    :param pdf: the list of pdf values
    :param x: a measured value
    :param u: a measurement uncertainty
    :param k: a coverage factor
    :param h: integration step
    """

    # suppose x in [0, 1]
    start_init = x - k * u
    start = max(start_init, 0.)
    end_init = x + k * u
    end = min(end_init, 1.)

    cut = "" if (isclose(start, start_init) and isclose(end, end_init)) else " (cut)"

    i_start = round(start / h)
    i_end = round(end / h)
    # a coverage probability for the symmetric (probably, cut) coverage interval
    prob_symm = round(np.trapz(pdf[i_start : i_end], dx=h), 3)
    p_0_perc = "{:.1f}%".format(100. * p_0)  # p_0 in %
    validity = f"  - Invalid! P < {p_0_perc}" if (prob_symm < p_0 - 1.e-3) else ""

    # to str
    start = ("{:." + str(n_digs) + "f}").format(start)
    end = ("{:." + str(n_digs) + "f}").format(end)
    prob_symm = "{:.1f}%".format(100. * prob_symm)

    interval = f"({start}, {end}), P = {prob_symm}"
    out = f"symmetric{cut} coverage interval ".ljust(INDENT) + interval + validity
    print(out)


def get_posterior_pdf(h: float, x: float, u: float, x_min: float, w: float, prior_tsp: bool) -> list:
    """
    Get a posterior pdf
    :param h: integration step
    :param x: a measured value
    :param u: a measurement uncertainty
    :param x_min: the expected lower border of the measurand
    :param w: a weight assigned to [x_min, 1] interval (a "degree of confidence" that x is actually in this range)
    :param prior_tsp: use asymmetric TSP distribution as a prior pdf if True; otherwise use uniform prior pdf
    :return: a list of posterior pdf values
    """

    n_vals = round(1. / h) + 1
    p = log(1. - w) / log(x_min)

    res = [0] * n_vals
    for i in range(n_vals):
        v = i * h
        if prior_tsp:
            res[i] = exp(-0.5 * ((v - x) / u) ** 2) * p * v ** (p - 1.)
        else:
            res[i] = 0 if v < x_min else exp(-0.5 * ((v - x) / u) ** 2)

    res = list(np.array(res) * (1. / np.trapz(res, dx=h)))  # norming
    return res


def format_parameter_output(param_name: str, value: float) -> str:
    """
    Format an output of the parameter value
    :param param_name: a name of the parameter (variable)
    :param value: the value
    :return: the formatted (justified) "name, value" string
    """
    formatted_value = ("{:." + str(n_digs) + "f}").format(value)
    return param_name.ljust(INDENT) + formatted_value


def main(x: float, u: float, x_min: float, w: float, k: float) -> None:
    """
    Perform calculations and put the results to console
    :param x: a measured value
    :param u: a measurement uncertainty
    :param x_min: the expected lower border of the measurand
    :param w: a weight assigned to [x_min, 1] interval (a "degree of confidence" that x is actually in this range)
    :param k: a coverage factor to be used to construct the symmetric coverage interval
    """

    # input values checks
    if not 0 <= x <= 1:
        raise ValueError(f"invalid x value: {x}")

    if not 0 <= x_min <= 1:
        raise ValueError(f"invalid c0 value: {x_min}")

    if u <= 0:
        raise ValueError(f"invalid (non-positive) uncertainty value: {u}")

    if not 0.5 <= w < 1:
        raise ValueError(f"invalid weight w value: {w}, the expected range is [0.5, 1)")

    if k <= 0:
        raise ValueError(f"invalid (non-positive) coverage factor value: {k}")

    # set number of decimal digits to perform the calculations

    n_digs_u = -floor(log10(u))
    h = 10 ** (-n_digs_u - 3)  # integration step value

    global n_digs
    n_digs = n_digs_u + 1

    print()
    print("input data")
    print("---------------------------------------------")
    for name, value in {"x": x, "u": u, "c0": x_min, "w": w}.items():
        print(format_parameter_output(name, value))
    print()

    alpha = (1. - x_min) / u
    beta = (1. - x) / u

    use_tsp_as_prior = (alpha <= 5) or (beta > alpha - 3)
    pdf = get_posterior_pdf(h, x, u, x_min, w, use_tsp_as_prior)
    assert isclose(h * sum(pdf), 1., abs_tol=0.01)  # check roughly the pdf normalization

    m = posterior_mean(pdf, h)
    sigma = posterior_stdev(pdf, m, h)

    max_val = max(pdf)
    mode = h * pdf.index(max_val)

    print("posterior pdf")
    print("---------------------------------------------")
    print(format_parameter_output("mean", m))
    print(format_parameter_output("mode", mode))
    print(format_parameter_output("stdev", sigma))

    shortest_coverage_interval(pdf, h)
    print_symmetric_coverage_interval(pdf, x, u, k, h)
    print()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--x", required=True, type=float, help="a measured value")
    parser.add_argument("--u", required=True, type=float, help="a reported measurement uncertainty")
    parser.add_argument("--c0", required=True, type=float,
                        help="a declared lower border for the measurand")
    parser.add_argument("--w", required=False, type=float, default=0.75,
                        help="a weight assigned to [x_min, 1] interval "
                             "(a \"degree of confidence\" that x is actually in this range)")
    parser.add_argument("--K", dest="k", required=False, type=float, default=2.,
                        help="a coverage factor value for the symmetric coverage interval")
    # TODO: make p_0 a parameter?

    args = parser.parse_args()
    main(args.x, args.u, args.c0, args.w, args.k)
