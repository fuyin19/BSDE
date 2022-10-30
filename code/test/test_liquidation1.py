import sys
sys.path.insert(0, '/Users/finn/Desktop/Capstone-BSDE/files/code')

import Liquidation1
import LSMC
import numpy as np


def test_liquidation1():
    """
    Test the PDE solver using the example in section 6.3
    """

    # Simulation parameters
    M = 2 ** 11
    N = 100
    T = 1
    d = 2
    dt = T / N
    dZ = LSMC.generate_z_matrix(n_paths=M, n_steps=N, d_bm=d)

    # Model parameters
    x0 = np.array([100, 0])   # S_0, q_0
    epsilon = 0.1
    k = 2

    X_sim = Liquidation1.X_t(sig_s=1, eps=epsilon)
    Y_sim = Liquidation1.Y_t(eps=epsilon, k=k)

    LSMC_solver = LSMC.LSMC_linear(Y_t=Y_sim, X_t=X_sim, x0=x0, dZ=dZ, dt=dt, basis_funcs_type='trig')
    LSMC_solver.solve()


def main():
    test_liquidation1()


if __name__ == '__main__':
    main()

