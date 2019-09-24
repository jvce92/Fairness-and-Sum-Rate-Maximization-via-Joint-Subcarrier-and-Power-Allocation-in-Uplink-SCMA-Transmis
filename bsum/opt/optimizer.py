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

from abc import ABC, abstractmethod
from collections import namedtuple

Status = namedtuple("Status", ["obj", "status", "variable"])


class BaseOptimizer(ABC):
    def __init__(self, K, J, **kwargs):
        self.K              = K
        self.J              = J
        self.max_it         = kwargs.get("max_it", 10)
        self.tol            = kwargs.get("tol", 1e-4)
        self.max_no_improve = kwargs.get("max_no_improve", 5)
        self.lam_val        = kwargs.get("lam", 50)
        self.max_joint_it   = kwargs.get("max_joint_it", 10)
        self.delta          = kwargs.get("delta", 5e-1)
        self.beta_succ      = kwargs.get("beta_succ", 1.5)
        self.beta_fail      = kwargs.get("beta_fail", 0.8)
        self.delta_cap      = kwargs.get("delta_cap", 2)
        self.alpha          = kwargs.get("alpha", 1)
        self.solver         = kwargs.get("solver", "ECOS")
        self.verbose        = kwargs.get("verbose", False)
        self.warmstart      = kwargs.get("warmstart", True)
        self.eps            = kwargs.get("epsilon", 1e-2)
        self.tau            = kwargs.get("tau", 1e-3)
        self.s              = self.tau / self.J
        self.niter = 0
        self.inner_iter = 0
        self.max_inner      = kwargs.get("max_inner", 100)

    def reset(self):
        self.niter = 0
        self.iter_no_improve = 0

    @abstractmethod
    def inner_step(self):
        raise NotImplementedError("Not implemented in base class")

    @abstractmethod
    def step(self):
        raise NotImplementedError("Not implemented in base class")

    @abstractmethod
    def solve(self):
        raise NotImplementedError("Not implemented in base class")