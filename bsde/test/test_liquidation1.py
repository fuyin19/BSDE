# import sys
# sys.path.insert(0, '/Users/finn/Desktop/Capstone-BSDE/files/bsde')

import numpy as np
from bsde.config import ConfigHJB1, ConfigLSMC
from bsde.solver.lsmc import LSMCLinear
from bsde.dynamics.liquidation1 import HJB_liquidation1_FBSDE


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

    # configg
    config_liquidation1 = ConfigHJB1(eps=epsilon, sig_s=sig_s, k=k, T=T, d=d, d1=d1, d2=d2)
    config_sim = ConfigLSMC(M=M, N=N, dt=dt, seed=42, x0=x0)

    # simulation
    FBSDE = HJB_liquidation1_FBSDE(config_liquidation1)
    LSMC_solver = LSMCLinear(FBSDE, config_sim, basis_funcs_type='trig')
    LSMC_solver.solve()


def main():
    test_liquidation1()


if __name__ == '__main__':
    main()

