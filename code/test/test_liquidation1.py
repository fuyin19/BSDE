# import sys
# sys.path.insert(0, '/Users/finn/Desktop/Capstone-BSDE/files/code')

import numpy as np
import code.config.config as cf
from code.LSMC import LSMC
from code.dynamics.Liquidation1 import HJB_liquidation1_FBSDE


def test_liquidation1():
    """
    Test the PDE solver using the example in section 6.3
    """

    # Simulation parameters
    M = 2 ** 11
    N = 100
    T = 1
    d = 2
    d1 = 2
    d2 = 1
    dt = T / N

    # Model parameters
    x0 = np.array([100, 0])   # S_0, q_0
    epsilon = 0.1
    sig_s = 0.5
    k = 2

    # config
    config_liquidation1 = cf.config_HJB_liquidation1(eps=epsilon, sig_s=sig_s, k=k, T=T, d=d, d1=d1, d2=d2)
    config_sim = cf.config_simulation(M=M, N=N, dt=dt, seed=42, x0=x0)

    # simulation
    FBSDE = HJB_liquidation1_FBSDE(config_liquidation1)
    LSMC_solver = LSMC.LSMC_linear(FBSDE, config_sim, basis_funcs_type='trig')
    LSMC_solver.solve()


def main():
    test_liquidation1()


if __name__ == '__main__':
    main()

