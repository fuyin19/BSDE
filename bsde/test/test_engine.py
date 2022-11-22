#import sys
#sys.path.insert(0, '/Users/finn/Desktop/Capstone-BSDE/files/bsde')

import numpy as np
import bsde.dynamics.european_option as eu

from bsde.config import ConfigOption, ConfigLSMC
from bsde.solver.lsmc import LSMCLinear
from bsde.engine import Engine


def test_engine():
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

    # Run the engine
    engine = Engine(FBSDEs, configs_LSMC_solver, LSMCLinear)
    engine.run()

    print(engine.res)


def main():
    test_engine()


if __name__ == '__main__':
    main()

