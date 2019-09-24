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

from .maxmin import JointMaxMinOptimizer
from .maxsr import JointMaxSROptimizer
