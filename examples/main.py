#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import argparse
import sys

import numpy as np
import prefopt
import prefopt.acquisition
import prefopt.optimization


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--acquisition", default="expected_improvement",
                        help="acquisition function")
    parser.add_argument("--optimizer", default="grid_search",
                        help="optimizer for acquisition function")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def display_result(x, qf, qgamma, precision=2, field_width=24):
    header = "|{:^" + str(field_width) + "s}|{:^" + str(field_width) + "s}|"
    print(header.format("X", "Q_f mean"))
    for a, b in zip(x, qf):
        fmt = "|{:>" + str(field_width) + "}|{:>" + str(field_width) + ".4f}|"
        print(fmt.format(np.array_str(a, precision=precision), b))
    # FIXME (jakob) print qgamma


def pref_func_eq_test(x1, x2):
    if abs(x1[0] - x2[0]) < 0.2:
        return 0
    else:
        return np.sign(np.array(x1)[0] - np.array(x2)[0])


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    # instantiate acquisition function
    if args.acquisition == "expected_improvement":
        acquisition_function = prefopt.acquisition.expected_improvement
    else:
        raise NotImplementedError("acquisition '{}'".format(args.acquisition))

    # instantiate optimizer
    if args.optimizer == "grid_search":
        optimizer = prefopt.optimization.grid_search
    else:
        raise NotImplementedError("optimizer '{}'".format(args.optimizer))

    # set up experiment
    n = 20.0
    domain_bb = [[0.0, n], [0.0, n]]
    exp = prefopt.Experiment(
        domain_bb,
        acquisition_function,
        optimizer
    )

    # add initial samples
    x1 = [1.0, 6.0]
    x2 = [5.0, 1.0]
    x3 = [14.0, 2.0]
    exp.add_preference(x1, x2, -1)
    exp.add_preference(x1, x3, -1)
    exp.add_preference(x2, x3, -1)

    # run experiment
    exp.setup_model()
    exp.run_inference()
    pref_func = pref_func_eq_test
    xb = x3
    n_iter = 3
    for i in xrange(n_iter):
        xn = exp.find_next(xb, verbose=args.verbose)
        # get preference input from user xn vs xb
        exp.add_preference(xb, xn, pref_func(xb, xn))
        if args.verbose:
            print(xn, " vs ", xb)
        if pref_func(xb, xn) == -1:
            xb = xn
        exp.setup_model()
        exp.run_inference()

    # print result
    if args.verbose:
        display_result(
            exp.X_np,
            exp.qf.mean().eval(),
            exp.qgamma.mean().eval()
        )


if __name__ == "__main__":
    sys.exit(main())
