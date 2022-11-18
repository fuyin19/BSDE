#import sys
#sys.path.insert(0, '/Users/finn/Desktop/Capstone-BSDE/files/bsde')

import numpy as np

from bsde.config.FBSDE import config_option
from bsde.config.simulation import config_simulation
from bsde.config.LSMC import config_linear, config_svm, config_NN

import bsde.dynamics.european_option as eu
from bsde.LSMC.LSMC import LSMC_linear
from bsde.LSMC.LSMC_engine import LSMC_engine


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
    configs_options = [config_option(r=r, sig=sig, K=K, T=T, d=d, d1=d1, d2=d2) for sig in sigs]
    configs_sim = [config_simulation(N=N, M=M, dt=dt, seed=42, x0=s0) for M in Ms]
    config_model = config_linear(model_params={})

    # dynamics
    FBSDEs = [eu.BS_FBSDE(cfg) for cfg in configs_options]

    # Run the engine
    engine = LSMC_engine(FBSDEs, configs_sim, config_model, LSMC_linear)
    engine.run()

    print(engine.res)


def main():
    test_engine()


if __name__ == '__main__':
    main()

