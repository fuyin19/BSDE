#import sys
#sys.path.insert(0, '/Users/finn/Desktop/Capstone-BSDE/files/bsde')

import numpy as np
import bsde.dynamics.european_option as eu

from bsde.config import ConfigOption, ConfigLSMC
from bsde.solver.lsmc import LSMCLinear
from bsde.engine import Engine
from bsde.test.demo import mc_european_call_cev


def test_engine(dynamics, configs_solver):
    # Run the engine
    engine = Engine(dynamics, configs_solver, LSMCLinear)
    engine.run()

    print(engine.res)


def test_mc_cev(cev_pde, cfgs_solver):
    beta = 0.9
    res = mc_european_call_cev(cev_pde, cfgs_solver, beta=beta)
    print(res)


def main():
    # Market Parameters
    r = 0.1
    sigs = [0.1, 0.5, 0.8]
    s0 = np.array([40])
    T = 1
    K = 30

    # Simulation parameters
    Ms = [int(2 ** i) for i in range(4, 19)]
    dt = (1 / 252.)
    N = int(T / dt)
    d = 1
    d1 = 1
    d2 = 1

    # configs
    configs_option = [ConfigOption(r=r, sig=sig, K=K, T=T, d=d, d1=d1, d2=d2) for sig in sigs]
    configs_LSMC_solver = [ConfigLSMC(N=N, M=M, dt=dt, seed=42, x0=s0, model_params={}, reg_method=None) for M in Ms]

    # dynamics
    FBSDEs = [eu.BS_FBSDE(cfg) for cfg in configs_option]

    # test_engine(FBSDEs, configs_LSMC_solver)
    test_mc_cev(FBSDEs, configs_LSMC_solver)


if __name__ == '__main__':
    main()

