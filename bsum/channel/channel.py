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

import numpy as np
from .geometry import random_geometry
from scipy.special import jv

J0 = lambda x : jv(0, x)

def channel_gain(K, J, R, alpha, ordered=True):
    r = random_geometry(J, R)
    # r = np.sort(r)
    path_loss = 1 + r**(alpha)
    H_t = (np.random.normal(0, 1/np.sqrt(2), size=(K, J)) +
           1j * np.random.normal(0, 1/np.sqrt(2), size=(K, J))) / np.sqrt(path_loss)
    if (ordered):
       H_t = np.array(sorted(H_t.T, key=lambda x: np.sum(np.abs(x)**2))).T
    H = np.real(np.conj(H_t) * H_t)
    # H = np.random.exponential(size=(K,J)) / path_loss

    return H, H_t

class GaussMarkovChannel:
    def __init__(self, K, J, R, alpha, max_doppler, delta_t):
       self.K = K
       self.J = J
       self.R = R
       self.corr = J0(2 * np.pi * delta_t * max_doppler)
       self.r = random_geometry(J, R)
       self.path_loss = 1 + self.r**(alpha)
       self.H_t = (np.random.normal(0, 1/np.sqrt(2), size=(K, J)) +
           1j * np.random.normal(0, 1/np.sqrt(2), size=(K, J))) / np.sqrt(self.path_loss)
       self.H_t = np.array(sorted(self.H_t.T, key=lambda x: np.sum(np.abs(x)**2))).T
       self.H = np.abs(self.H_t)**2

    def __call__(self):
        return self.H

    def __next__(self):
        innovation = (np.random.normal(0, np.sqrt(1 - self.corr**2), size=(self.K, self.J)) + 
                      1j * np.random.normal(0, np.sqrt(1 - self.corr**2), size=(self.K, self.J))) / np.sqrt(self.path_loss)
        self.H_t = self.corr * self.H_t + innovation
        self.H = np.abs(self.H_t)**2

        return self.H

    def __iter__(self):
        return self


