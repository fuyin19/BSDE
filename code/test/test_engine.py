#import sys
#sys.path.insert(0, '/Users/finn/Desktop/Capstone-BSDE/files/code')

import code.config.config as cf
import numpy as np
import code.dynamics.european_option as eu
from code.LSMC.LSMC_engine import LSMC_engine


def test_engine():
    # Market Parameters
    r = 0.1
    sigs = [0.1*i for i in range(1, 2)]
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
    configs_options = [cf.config_option(r=r, sig=sig, K=K, T=T, d=d, d1=d1, d2=d2) for sig in sigs]
    configs_sim = [cf.config_simulation(N=N, M=M, dt=dt, seed=42, x0=s0) for M in Ms]

    # dynamics
    FBSDEs = [eu.BS_FBSDE(cfg) for cfg in configs_options]

    # Run the engine
    engine = LSMC_engine(FBSDEs, configs_sim)
    engine.run()

    print(engine.res)

def main():
    test_engine()


if __name__ == '__main__':
    main()

