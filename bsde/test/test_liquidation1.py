# import sys
# sys.path.insert(0, '/Users/finn/Desktop/Capstone-BSDE/files/bsde')

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from bsde.config import ConfigHJB1, ConfigLSMC, ConfigDeepBSDE, ConfigFBSNN
from bsde.solver.lsmc import LSMCLinear
from bsde.solver.deep_bsde import DeepBSDESolver
from bsde.dynamics.liquidation1 import HJB_liquidation1_FBSDE, HJB_liquidation1_solver


def test_liquidation1_lsmc(cfg_pde, cfg_lsmc_solver):
    """
    Test the PDE solver using the example in section 6.3
    """
    FBSDE = HJB_liquidation1_FBSDE(config=cfg_pde, exclude_spot=True)
    LSMC_solver = LSMCLinear(FBSDE, cfg_lsmc_solver, basis_funcs_type='trig')
    LSMC_solver.solve()


def test_liquidation1_deep(cfg_pde, cfg_deep_solver):
    FBSDE = HJB_liquidation1_FBSDE(config=cfg_pde, exclude_spot=True)
    deep_solver = DeepBSDESolver(FBSDE, cfg_deep_solver)
    deep_solver.train()


def test_liquidation1_FBSNN(cfg_pde, cfg_FBSNN):
    HJB_FBSNN = HJB_liquidation1_solver(config_dynamic=cfg_pde, config_solver=cfg_FBSNN)
    HJB_FBSNN.train(N_Iter=3 * 10 ** 4, learning_rate=7e-3)
    HJB_FBSNN.train(N_Iter=4 * 10 ** 3, learning_rate=2e-3)
    HJB_FBSNN.train(N_Iter=3 * 10 ** 3, learning_rate=1e-4)


def main():
    # Simulation parameters
    T = 0.25
    dt = 1 / 252
    N = int(T/dt)
    d = 2
    d1 = 2
    d2 = 1
    seed = 42

    # Model parameters
    x0 = np.array([30., 10000.])   # S_0, q_0
    epsilon = 0.001
    sig_s = 0.5
    lb = 1.
    k = 0.001

    print('Analytical Y0: {}'.format(x0[0]*x0[1] - x0[1]**2 / (1/lb + T/k)))

    # configg
    cfg_HJB1 = ConfigHJB1(sig_s=sig_s, eps=epsilon, lb=lb, k=k, T=T, d=d, d1=d1, d2=d2)

    # FBSNN
    if True:
        tf.compat.v1.disable_v2_behavior()
        batch_size = 1
        cfg_FBSNN = ConfigFBSNN(x0=x0.reshape(1, d), N=N, M=batch_size, dt=dt, seed=seed, layers=[d+1] + 4*[256] + [1])
        test_liquidation1_FBSNN(cfg_pde=cfg_HJB1, cfg_FBSNN=cfg_FBSNN)

    if False:
        # LSMC
        M = 2 ** 16  # N_path
        cfg_lsmc_linear = ConfigLSMC(M=M, N=N, dt=dt, seed=42, x0=x0,
                                     model_params={'fit_intercept': False}, basis_funcs_type='poly')
        test_liquidation1_lsmc(cfg_HJB1, cfg_lsmc_linear)

    if False:
        # deep BSDE
        tf.compat.v1.enable_v2_behavior()
        batch_size = 64
        cfg_deep_solver = ConfigDeepBSDE(N=N, dt=dt, seed=42, x0=x0,
                                         y_init_range=[-98000, -98100],
                                         n_hiddens=[10+d, 10+d],
                                         lr_values=[2e-3, 1e-3],
                                         lr_boundaries=[6000],
                                         n_iterations=9000,
                                         batch_size=batch_size,
                                         valid_size=64,
                                         report_freq=100,
                                         dtype='float64',
                                         verbose=True)

        test_liquidation1_deep(cfg_HJB1, cfg_deep_solver)


if __name__ == '__main__':
    main()

