''' Created on Tue Sep 24 2019

GNU General Public License
Copyright (c) 2019 Joao V.C. Evangelista

This code is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This code is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

If you use this code or a modified version, we kindly request you
to also cite our original work accepted for publication at
IEEE Transactions on Wireless Communications (https://ieeexplore.ieee.org/document/8836645).

Bibtex:
@ARTICLE{8836645,
author={J. V. C. {Evangelista} and Z. {Sattar} and G. {Kaddoum} and A. {Chaaban}},
journal={IEEE Transactions on Wireless Communications},
title={Fairness and Sum-Rate Maximization via Joint Subcarrier and Power Allocation in Uplink SCMA Transmission},
year={2019},
volume={},
number={},
pages={1-1},
keywords={Resource management;Approximation algorithms;NOMA;Optimization;Uplink;Heuristic algorithms;Decoding;SCMA;5G;Power Allocation;Multiple Access},
doi={10.1109/TWC.2019.2939820},
ISSN={},
month={},}
'''

import argparse
from datetime import datetime
from functools import reduce
from pathlib import Path
from sys import argv, stderr

import numpy as np
from pathos.pools import ProcessPool
from tqdm import tqdm

from bsum.channel.channel import GaussMarkovChannel, channel_gain
from bsum.misc.jain import jain
from bsum.misc.rate import rate
from bsum.opt import JointMaxMinOptimizer, JointMaxSROptimizer


class PerfRunner:
    def __init__(self, K, J, N, df, alpha, radius, N0, algo, solver, print_every):
        self.K = K
        self.J = J
        self.N = N
        self.df = df
        self.alpha = alpha
        self.radius = radius
        self.N0 = N0
        self.solver = solver
        self.algo = algo
        self.print_every = print_every

    def __call__(self, pm, it):
        return run_perf(self.K,
                        self.J,
                        self.N,
                        self.df,
                        self.alpha,
                        self.radius,
                        self.N0,
                        pm,
                        it,
                        self.algo,
                        self.solver,
                        self.print_every)


class OutdatedRunner:
    def __init__(self,
                 K,
                 J,
                 N,
                 df,
                 alpha,
                 radius,
                 N0,
                 Pmax,
                 doppler,
                 delta_t,
                 algo,
                 solver,
                 print_every):
        self.K = K
        self.J = J
        self.N = N
        self.df = df
        self.alpha = alpha
        self.radius = radius
        self.N0 = N0
        self.Pmax = Pmax
        self.doppler = doppler
        self.delta_t = delta_t
        self.solver = solver
        self.algo = algo
        self.print_every = print_every

    def __call__(self, update_freq, it):
        return run_outdated(K=self.K,
                            J=self.J,
                            N=self.N,
                            df=self.df,
                            alpha=self.alpha,
                            radius=self.radius,
                            N0=self.N0,
                            Pmax=self.Pmax,
                            doppler=self.doppler,
                            delta_t=self.delta_t,
                            update_freq=update_freq,
                            it=it,
                            algo=self.algo,
                            solver=self.solver,
                            print_every=self.print_every)


def run_outdated(K,
                 J,
                 N,
                 df,
                 alpha,
                 radius,
                 N0,
                 Pmax,
                 doppler,
                 delta_t,
                 update_freq,
                 it,
                 algo,
                 solver,
                 print_every):
    done = False
    niter = 0

    while not done:
        channel = GaussMarkovChannel(K, J, R, alpha, doppler, delta_t)
        H = channel()
        optimizer = algorithms[algo](K, J, solver=solver)
        optimizer.reset(N, df, Pmax, H, N0)

        try:
            F, P = optimizer.solve()
            F = np.round(F)
            P = F * P
        except Exception as e:
            print("Exception thrown during UF={0} iteration {1}! {2}".format(
                    update_freq, it, e), file=stderr)
            continue

        if np.all(F==0) or np.all(P==0):
            continue

        done = True

    user_rate = np.sum(rate(F, P, H, N0), axis=0)
    jain_idx = jain(user_rate) / update_freq
    sum_rate = np.sum(user_rate) / update_freq

    for n in range(update_freq-1):
        H = next(channel)
        user_rate = np.sum(rate(F, P, H, N0), axis=0)
        jain_idx += jain(user_rate) / update_freq
        sum_rate += np.sum(user_rate) / update_freq

    return (update_freq, sum_rate, jain_idx)


def run_perf(K,
             J,
             N,
             df,
             alpha,
             radius,
             N0,
             Pmax,
             it,
             algo,
             solver,
             print_every):
    done = False
    while not done:
        optimizer = algorithms[algo](K, J, solver=solver)
        H, _ = channel_gain(K, J, R, alpha)
        optimizer.reset(N, df, Pmax, H, N0)

        try:
            F, P = optimizer.solve()
            F = np.round(F)
            P = F * P
            user_rate = np.sum(rate(F, P, H, N0), axis=0)
            jain_idx = jain(user_rate)
            sum_rate = np.sum(user_rate)
        except Exception as e:
            print("Exception thrown during Pmax={0} iteration {1}! {2}".format(
                Pmax, it, e), file=stderr)
            continue

        # if it % print_every:
        #     print("Iteration #{0}, Pmax = {1}".format(it, Pmax))

        done = True

    return (Pmax, sum_rate, jain_idx)


def gen_inputs(var, num_iter):
    return [(v, it) for v in var for it in np.arange(num_iter)]


def reduce_outputs(var, num_iter, results):
    sum_rate = np.zeros_like(var, dtype=np.float)
    jain_idx = np.zeros_like(var, dtype=np.float)

    for idx, v in enumerate(var):
        filt = list(filter(lambda x: x[0] == v, results))
        sum_rate[idx] = reduce(lambda x, y: x+y, list(zip(*filt))[1])/num_iter
        jain_idx[idx] = reduce(lambda x, y: x+y, list(zip(*filt))[2])/num_iter

    return sum_rate, jain_idx

class StoreNPArray(argparse._StoreAction):
    def __call__(self, parser, namespace, values, option_string=None):
        values = np.array(values)
        return super().__call__(parser, namespace, values, option_string)


parser = argparse.ArgumentParser(prog="python3 run.py")
parser.add_argument(
    "-K", "--num-sc",
    type=int,
    default=4,
    help="Number of subcarriers available"
)
parser.add_argument(
    "-J", "--num-users",
    type=int,
    default=6,
    help="Number of SCMA layers"
)
parser.add_argument(
    "-Pm", "--Pmax",
    type=float,
    action=StoreNPArray,
    default=np.linspace(3, 10, 8),
    nargs="+",
    help="Array with power allocation budget in dBm"
)
parser.add_argument(
    "-N", "--num-sc-per-user",
    type=int,
    default=2,
    help="Maximum number of SCMA layers allocated per user"
)
parser.add_argument(
    "-df", "--num-users-per-sc",
    type=int,
    default=3,
    help="Maximum number of users allocated per SCMA layers"
)
parser.add_argument(
    "-Np", "--num-procs",
    type=int,
    default=4,
    help="Number of process to run simulation on"
)
parser.add_argument(
    "-bw", "--bandwidth",
    type=float,
    default=180e3,
    help="Bandwidth in Hertz"
)
parser.add_argument(
    "-N0", "--noise-power",
    type=float,
    default=-174,
    help="Noise power in dB/Hz"
)
parser.add_argument(
    "-r", "--radius",
    type=float,
    default=300,
    help="Wireless cell radius"
)
parser.add_argument(
    "-a", "--alpha",
    type=float,
    default=4,
    help="Path loss exponent"
)
parser.add_argument(
    "-dpath", "--data-path",
    type=str,
    default=".",
    help="Path to store simulation data"
)
parser.add_argument(
    "-algo", "--algorithm",
    choices=["maxmin", "maxsr"],
    default="maxmin",
    help="Optimization algorithm to choose: maxmin (Fair allocation) or maxsr (Maximize sum-rate)"
)
parser.add_argument(
    "-niter", "--num-iter",
    type=int,
    default=1000,
    help="Number of samples for Monte Carlo simulation"
)
parser.add_argument(
    "-p", "--print-every",
    type=int,
    default=100,
    help="Print update after every iterations"
)
parser.add_argument(
    "-s", "--solver",
    type=str,
    default="ECOS",
    choices=["ECOS", "SCS", "MOSEK"],
    help="Choice of conic solver supporting exponential cones"
)
parser.add_argument(
    "-uf", "--update-freq",
    type=int,
    default=np.linspace(1, 50, 16, dtype=np.int),
    action=StoreNPArray,
    nargs="+",
    help="Array with update frequencies per pilot transmission"
)
parser.add_argument(
    "-t", "--delta_t",
    type=float,
    default=1e-2,
    help="Time interval between successive transmission blocks"
)
parser.add_argument(
    "-dop", "--doppler-freq",
    type=float,
    default=10,
    help="Maximum Doppler frequence"
)
parser.add_argument(
    "-prob", "--problem",
    type=str,
    choices=["outdated", "perf"],
    default="perf",
    help="Run script for either evaluate algorithm performance or the performance \
          under outdated CSI"
)

algorithms = {
    "maxsr": JointMaxSROptimizer,
    "maxmin": JointMaxMinOptimizer
}


if __name__ == "__main__":
    print("Starting cluster job")

    args = parser.parse_args(argv[1:])
    K = args.num_sc
    J = args.num_users
    Pmax = 1e-3 * 10**(args.Pmax / 10)
    N0 = args.bandwidth * 10**(args.noise_power/10)
    df = args.num_users_per_sc
    N = args.num_sc_per_user
    optimizer = algorithms[args.algorithm]
    num_iter = args.num_iter
    doppler = args.doppler_freq
    update_freq = args.update_freq
    delta_t = args.delta_t
    alpha = args.alpha
    solver = args.solver
    R = args.radius
    data_path = Path(args.data_path)
    algorithm = args.algorithm
    print_every = args.print_every

    perf_call = PerfRunner(K, J, N, df, alpha, R, N0,
                           algorithm, solver, print_every)
    outdated_call = OutdatedRunner(K, J, N, df, alpha, R, N0,
                                   Pmax[-1], doppler, delta_t, algorithm,
                                   solver, print_every)

    if args.problem == "perf":
        print("Starting performance simulation:")
        print("Pmax = {0}".format(Pmax.tolist()))
        var = Pmax
        sim_call = perf_call
    elif args.problem == "outdated":
        print("Starting outdated CSI simulation:")
        print("Update frequencies = {0}".format(update_freq.tolist()))
        var = update_freq
        sim_call = outdated_call

    inputs = gen_inputs(var, num_iter)

    pool = ProcessPool(args.num_procs)
    results = []
    for r in tqdm(pool.imap(sim_call, *zip(*inputs)), total=len(var)*num_iter):
        results.append(r)
    pool.close()
    sum_rate, jain_idx = reduce_outputs(var, num_iter, results)

    print("Job done!")

    date = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    dir_name = "-".join([algorithm, date, solver, args.problem, "R" + str(R)])
    if args.problem == "outdated":
        dir_name += "-doppler" + str(doppler)

    if not data_path.exists():
        print("Creating data directory")
        data_path.mkdir()
    if not data_path.joinpath(dir_name).exists():
        print("Creating simulation data")
        data_path.joinpath(dir_name).mkdir()

    np.save(data_path.joinpath(dir_name).joinpath("sum_rate"), sum_rate)
    np.save(data_path.joinpath(dir_name).joinpath("jain"), jain_idx)

    print("Data saved successfully")
