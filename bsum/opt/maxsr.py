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

import cvxpy as cvx
import numpy as np
from matplotlib import pyplot as plt

from ..misc import jain, rate
from .optimizer import BaseOptimizer, Status

default_params = {"max_it": 10,
                  "tolerance": 1e-4,
                  "max_no_improve": 10,
                  "lambda": 50,
                  "max_joint_it": 5,
                  "pwr_scaling": 1e-3,
                  "ch_scaling": 1,
                  "alpha": 1,
                  "F_perturb": 1e-2,
                  "P_perturb": 1e-4,
                  "delta_F": 5e-1,
                  "delta_P": 1e-2,
                  "beta_F_succ": 1.5,
                  "beta_F_fail": 0.8,
                  "delta_F_cap": 1,
                  "alpha_F": 1.1,
                  "beta_P_succ": 1.5,
                  "beta_P_fail": 0.8,
                  "delta_P_cap": 1e-1,
                  "alpha_P": 1.1, }
solver = "ECOS"


class MaxSRPowerOptimizer(BaseOptimizer):
    def __init__(self, K, J, **kwargs):
        super().__init__(K, J, **kwargs)
        self.P = cvx.Variable((self.K, self.J), nonneg=True)
        self.Fk = cvx.Parameter((self.K, self.J), nonneg=True)
        # self.Pk = cvx.Parameter((self.K, self.J), nonneg=True)
        self.Pmax = cvx.Parameter(nonneg=True)
        self.rho = cvx.Parameter((self.K, self.J), nonneg=True)

        self.obj = 0
        constraints = []#[cvx.norm(self.P - self.Pk, 1) <= self.delta]

        for k in range(self.K):
            num = 0
            for j in range(self.J):
                num += self.rho[k, j] * self.P[k, j] * self.Fk[k, j]
            self.obj += cvx.log(1 + num) - (cvx.max(self.rho[k, :]) - cvx.min(self.rho[k, :]))/2

        for j in range(self.J):
            constraints.append(
                self.P[:, j] * self.Fk[:, j] <= self.Pmax
            )
            constraints.append(
                self.P[:, j] * self.Fk[:, j] >= 0
            )
            
        self.prob = cvx.Problem(cvx.Maximize(self.obj), constraints)

    def reset(self, Pk, Fk, Pmax, H, N0):
        super().reset()
        self.obj_prev = 0
        self.obj_val = 1
        self.Pk_prev = np.zeros((self.K, self.J))
        self.H = H
        self.N0 = N0
        # self.Pk.value = self.Pk.project(Pk)
        self.Fk.value = self.Fk.project(Fk)
        self.Pmax.value = self.Pmax.project(Pmax)
        self.rho.value = self.rho.project(H/N0)
        self.inner_iter = 0

    def inner_step(self):
        try:
            if self.solver == "SCS":
                self.prob.solve(solver=self.solver,
                                verbose=self.verbose,
                                warm_start=self.warmstart,
                                max_iters=500,
                                gpu=True)
            else:
                self.prob.solve(solver=self.solver,
                                verbose=self.verbose, warm_start=self.warmstart)
        except cvx.SolverError as e:
            print("Solver error! Returning best iterate (Power iteration)")

        # self.Pk_prev = self.Pk.value
        # self.Pk.value = self.Pk.project(self.P.value)
        self.obj_prev = self.obj_val
        self.obj_val = self.obj.value

        if self.obj_val < self.obj_prev:
            self.iter_no_improve += 1

        return self.prob.status

    def step(self):
        while self.niter < self.max_it and np.abs(self.obj_val - self.obj_prev) > self.tol and self.iter_no_improve < self.max_no_improve:
            self.niter += 1
            status = self.inner_step()
            if status not in ['optimal', 'optimal_inaccurate']:
                raise RuntimeError(
                    "Power optimizer did not converge! Status: {0}".format(status))

            if self.prob.solver_stats.num_iters is not None:
                self.inner_iter += self.prob.solver_stats.num_iters

        return Status(obj=self.obj_val, status=status, variable=self.P.value)

    def solve(self):
        raise NotImplementedError("Not implemented in the power optimizer")


class MaxSRSubcarrierOptimizer(BaseOptimizer):
    def __init__(self, K, J, **kwargs):
        super().__init__(K, J, **kwargs)
        self.F = cvx.Variable((self.K, self.J), nonneg=True)
        self.Fk = cvx.Parameter((self.K, self.J), nonneg=True)
        self.Pk = cvx.Parameter((self.K, self.J), nonneg=True)
        self.Pmax = cvx.Parameter(nonneg=True)
        self.N = cvx.Parameter(nonneg=True)
        self.df = cvx.Parameter(nonneg=True)
        self.rho = cvx.Parameter((self.K, self.J), nonneg=True)
        self.lam = cvx.Parameter(nonneg=True)

        self.obj = 0
        for k in range(self.K):
            num = 0
            for j in range(self.J):
                num += self.rho[k, j] * self.Pk[k, j] * self.F[k, j]
            self.obj += cvx.log(1 + num) - (cvx.max(self.rho[k, :]) - cvx.min(self.rho[k, :]))/2
                
        constraints = [#cvx.norm(self.F - self.Fk, 1) <= self.delta,
                       cvx.sum(self.F, axis=0) <= self.N,
                       cvx.sum(self.F, axis=0) >= 1,
                       cvx.sum(self.F, axis=1) <= self.df,
                       self.F <= 1]
        for j in range(self.J):
            for k in range(self.K):
                self.obj += self.lam * ((self.Fk[k, j]**2 - self.Fk[k, j]) + (
                    2 * self.Fk[k, j] - 1) * (self.F[k, j] - self.Fk[k, j]))
            constraints.append(
                self.Pk[:, j] * self.F[:, j] <= self.Pmax
            )
            constraints.append(
                self.Pk[:, j] * self.F[:, j] >= 0
            )
            
        self.prob = cvx.Problem(cvx.Maximize(self.obj), constraints)

    def reset(self, Pk, Fk, N, df, Pmax, H, N0):
        super().reset()
        self.obj_prev = 0
        self.obj_val = 1
        self.Fk_prev = np.zeros((self.K, self.J))
        self.H = H
        self.N0 = N0
        self.Pk.value = self.Pk.project(Pk)
        self.Fk.value = self.Fk.project(Fk)
        self.Pmax.value = self.Pmax.project(Pmax)
        self.N.value = self.N.project(N)
        self.df.value = self.df.project(df)
        self.rho.value = self.rho.project(H/N0)
        self.lam.value = self.lam.project(self.lam_val)
        self.inner_iter = 0

    def inner_step(self):
        try:
            if self.solver == "SCS":
                self.prob.solve(solver=self.solver,
                                verbose=self.verbose,
                                warm_start=self.warmstart,
                                max_iters=500,
                                gpu=True)
            else:
                self.prob.solve(solver=self.solver,
                                verbose=self.verbose, warm_start=self.warmstart)

        except cvx.SolverError as e:
            print("Solver error! Returning best iterate (Subcarrier iteration)")

        phi_prev = np.sum(rate(self.Fk_prev, self.Pk.value, self.H, self.N0)) +\
                     self.lam.value * np.sum(self.Fk_prev**2 - self.Fk_prev)
        phi_new = np.sum(rate(self.F.value, self.Pk.value, self.H, self.N0)) + \
                     self.lam.value * np.sum(self.F.value**2 - self.F.value)
        sigma_hat = self.obj.value - phi_prev
        sigma = phi_new - phi_prev

        if (sigma >= self.alpha * sigma_hat):
            self.delta = np.minimum(
                self.delta_cap, self.beta_succ * self.delta)
            self.Fk_prev = self.Fk.value
            self.Fk.value = self.Fk.project(self.F.value)
        else:
            self.iter_no_improve += 1
            self.delta = self.beta_fail * self.delta
        self.obj_val = self.obj.value

        return self.prob.status

    def step(self):
        while self.niter < self.max_it and \
                np.abs(self.obj_val - self.obj_prev) > self.tol and \
                self.iter_no_improve < self.max_no_improve:
            self.niter += 1
            status = self.inner_step()
            if status not in ['optimal', 'optimal_inaccurate']:
                raise RuntimeError(
                    "Subcarrier optimizer did not converge! Status: {0}".format(status))

            if self.lam.value * np.sum(self.Fk.value**2 - self.Fk.value) < self.tol:
                break

            if self.prob.solver_stats.num_iters is not None:
                self.inner_iter += self.prob.solver_stats.num_iters

        return Status(obj=self.obj_val, status=status, variable=self.Fk.value)

    def solve(self):
        raise NotImplementedError(
            "Not implemented in the subcarrier optimizer")


class JointMaxSROptimizer:
    def __init__(self, K, J, **kwargs):
        max_it = kwargs.get("max_it", 10)
        pwr_tol = kwargs.get("pwr_tol", 1e-4)
        sc_tol = kwargs.get("sc_tol", 1e-4)
        max_no_improve = kwargs.get("max_no_improve", 5)
        lam = kwargs.get("lam", 50)
        delta_F = kwargs.get("delta_F", 5e-1)
        beta_F_succ = kwargs.get("beta_F_succ", 1.5)
        beta_F_fail = kwargs.get("beta_F_fail", 0.8)
        delta_F_cap = kwargs.get("delta_F_cap", 2)
        alpha_F = kwargs.get("alpha_F", 1)
        delta_P = kwargs.get("delta_P", 5e-1)
        beta_P_succ = kwargs.get("beta_P_succ", 1.5)
        beta_P_fail = kwargs.get("beta_P_fail", 0.8)
        delta_P_cap = kwargs.get("delta_P_cap", 2)
        alpha_P = kwargs.get("alpha_P", 1)
        solver = kwargs.get("solver", "ECOS")
        verbose = kwargs.get("verbose", False)
        warmstart = kwargs.get("warmstart", True)

        self.K = K
        self.J = J
        self.max_it = kwargs.get("max_joint_it", 5)
        self.tol = kwargs.get("joint_tol", 1e-4)

        self.Fk_prev = np.zeros((K, J))
        self.Pk_prev = np.zeros((K, J))

        self.pwr_opt = MaxSRPowerOptimizer(
            K, J,
            max_it=max_it,
            tol=pwr_tol,
            max_no_improve=max_no_improve,
            delta=delta_P,
            beta_succ=beta_P_succ,
            beta_fail=beta_P_fail,
            delta_cap=delta_P_cap,
            alpha=alpha_P,
            solver=solver,
            verbose=verbose,
            warmstart=warmstart
        )

        self.sc_opt = MaxSRSubcarrierOptimizer(
            K, J,
            max_it=max_it,
            tol=sc_tol,
            max_no_improve=max_no_improve,
            delta=delta_F,
            beta_succ=beta_F_succ,
            beta_fail=beta_F_fail,
            delta_cap=delta_F_cap,
            alpha=alpha_F,
            lam=lam,
            solver=solver,
            verbose=verbose,
            warmstart=warmstart
        )

    def reset(self, N, df, Pmax, H, N0):
        self.Pmax = Pmax
        self.H = H
        self.N0 = N0
        self.N = N
        self.df = df
        self.niter = 0
        self.Fk = np.ones((self.K, self.J)) * np.min([df/self.J, N/self.K])
        self.Pk = np.ones((self.K, self.J)) * Pmax/self.K
        self.pwr_opt.reset(self.Pk, self.Fk, Pmax, H, N0)
        self.sc_opt.reset(self.Pk, self.Fk, N, df, Pmax, H, N0)

    def power_iteration(self):
        self.pwr_opt.reset(self.Pk, self.Fk, self.Pmax, self.H, self.N0)
        status = self.pwr_opt.step()
        self.Pk_prev = self.Pk
        self.Pk = status.variable

    def subcarrier_iteration(self):
        self.sc_opt.reset(self.Pk, self.Fk, self.N, self.df,
                          self.Pmax, self.H, self.N0)
        status = self.sc_opt.step()
        self.Fk_prev = self.Fk
        self.Fk = status.variable

    def solve(self):
        while (np.linalg.norm(self.Pk * self.Fk - self.Pk_prev * self.Fk_prev) > self.tol or
               np.linalg.norm(self.Fk-self.Fk_prev) > self.tol) and \
                self.niter < self.max_it or self.niter <= 3:
            # print(np.linalg.norm(Pk * Fk - Pk_prev * Fk_prev))
            self.niter += 1
            self.subcarrier_iteration()
            self.power_iteration()

        return self.Fk, self.Pk
