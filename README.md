# Abstract

In this work, we consider a sparse code multiple access uplink system, where J users simultaneously transmit data over K subcarriers, such that J > K, with a constraint on the power transmitted by each user. To jointly optimize the subcarrier assignment and the transmitted power per subcarrier, two new iterative algorithms are proposed, the first one aims to maximize the sum-rate (Max-SR) of the network, while the second aims to maximize the fairness (Max-Min). In both cases, the optimization problem is of the mixed-integer nonlinear programming (MINLP) type, with non-convex objective functions, which are generally not tractable. We prove that both joint allocation problems are NP-hard. To address these issues, we employ a variant of the block successive upper-bound minimization (BSUM) 1 framework, obtaining polynomial-time approximation algorithms to the original problem. Moreover, we evaluate the algorithms’ robustness against outdated channel state information (CSI), present an analysis of the convergence of the algorithms, and a comparison of the sum-rate and Jain’s fairness index of the novel algorithms with three other algorithms proposed in the literature. The Max-SR algorithm outperforms the others in the sum-rate sense, while the Max-Min outperforms them in the fairness sense.

[Link to paper](https://ieeexplore.ieee.org/document/8836645)

# Acknowledgement

This work was also supported by the Tier 2 Canada Research Chair on the  Next Generations of  Wireless IoT Networks

My Ph.D. is funded by the [Mitacs Globalink Graduate Fellowship](https://www.mitacs.ca/) and the [FRQNT B2X Doctoral Fellowship](http://www.frqnt.gouv.qc.ca/).

# Usage

## Installing Dependencies

### pip

```bash
pip install -r requirements.txt
```

### Anaconda

Coming soon...

## Running

The simulation can be run with the script `run.py` with the options:

```
usage: python3 run.py [-h] [-K NUM_SC] [-J NUM_USERS] [-Pm PMAX [PMAX ...]]
                      [-N NUM_SC_PER_USER] [-df NUM_USERS_PER_SC]
                      [-Np NUM_PROCS] [-bw BANDWIDTH] [-N0 NOISE_POWER]
                      [-r RADIUS] [-a ALPHA] [-dpath DATA_PATH]
                      [-algo {maxmin,maxsr}] [-niter NUM_ITER]
                      [-p PRINT_EVERY] [-s {ECOS,SCS,MOSEK}]
                      [-uf UPDATE_FREQ [UPDATE_FREQ ...]] [-t DELTA_T]
                      [-dop DOPPLER_FREQ] [-prob {outdated,perf}]

optional arguments:
  -h, --help            show this help message and exit
  -K NUM_SC, --num-sc NUM_SC
                        Number of subcarriers available
  -J NUM_USERS, --num-users NUM_USERS
                        Number of SCMA layers
  -Pm PMAX [PMAX ...], --Pmax PMAX [PMAX ...]
                        Array with power allocation budget in dBm
  -N NUM_SC_PER_USER, --num-sc-per-user NUM_SC_PER_USER
                        Maximum number of SCMA layers allocated per user
  -df NUM_USERS_PER_SC, --num-users-per-sc NUM_USERS_PER_SC
                        Maximum number of users allocated per SCMA layers
  -Np NUM_PROCS, --num-procs NUM_PROCS
                        Number of process to run simulation on
  -bw BANDWIDTH, --bandwidth BANDWIDTH
                        Bandwidth in Hertz
  -N0 NOISE_POWER, --noise-power NOISE_POWER
                        Noise power in dB/Hz
  -r RADIUS, --radius RADIUS
                        Wireless cell radius
  -a ALPHA, --alpha ALPHA
                        Path loss exponent
  -dpath DATA_PATH, --data-path DATA_PATH
                        Path to store simulation data
  -algo {maxmin,maxsr}, --algorithm {maxmin,maxsr}
                        Optimization algorithm to choose: maxmin (Fair
                        allocation) or maxsr (Maximize sum-rate)
  -niter NUM_ITER, --num-iter NUM_ITER
                        Number of samples for Monte Carlo simulation
  -p PRINT_EVERY, --print-every PRINT_EVERY
                        Print update after every iterations
  -s {ECOS,SCS,MOSEK}, --solver {ECOS,SCS,MOSEK}
                        Choice of conic solver supporting exponential cones
  -uf UPDATE_FREQ [UPDATE_FREQ ...], --update-freq UPDATE_FREQ [UPDATE_FREQ ...]
                        Array with update frequencies per pilot transmission
  -t DELTA_T, --delta_t DELTA_T
                        Time interval between successive transmission blocks
  -dop DOPPLER_FREQ, --doppler-freq DOPPLER_FREQ
                        Maximum Doppler frequence
  -prob {outdated,perf}, --problem {outdated,perf}
                        Run script for either evaluate algorithm performance
                        or the performance under outdated CSI
```

## Solvers

We make use of the [cvxpy](https://github.com/cvxgrp/cvxpy) module that supports the solvers in this [link](https://www.cvxpy.org/tutorial/advanced/index.html#setting-solver-options). The script has been tested with [ecos](https://github.com/embotech/ecos), [scs](https://github.com/cvxgrp/scs) and [Mosek](https://www.mosek.com/) (starting from version 9 with support for exponential cones). 

We achieved the best results in running time and quality of the solutions found using the Mosek solver, however the software requires a license (that can be obtained for free for students). Among the open source solvers ecos had better results.