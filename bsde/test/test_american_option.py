# import sys
# sys.path.insert(0, '/Users/finn/Desktop/Capstone-BSDE/files/bsde')

import numpy as np

from bsde.dynamics import american_option, european_option
from bsde.config import ConfigOption, ConfigLSMC, ConfigDeepBSDE
from bsde.solver.lsmc import LSMCLinear
from bsde.solver.deep_bsde import DeepBSDESolver
from bsde.test.demo import show_bs_pde_result


def test_american_put(payoff_type, cfg_am_put, cfg_lsmc_solver, cfg_deep_solver, **kwargs):
    """
    Test the american put option price computed by dynamics
    """
    # Option type
    if payoff_type == 'vanilla':
        FBSDE = american_option.FSBDEAmericanPut(cfg_am_put)
    elif payoff_type == 'barrier':
        FBSDE = american_option.FBSDEAmericanPutBarrier(cfg_am_put, lower_barrier=kwargs.get('lower_barrier'))

    # LSMC Method 1
    LSMC_solver = LSMCLinear(FBSDE, cfg_lsmc_solver, basis_funcs_type='poly')
    LSMC_solver.solve()
    print("American {} option pricing by BSDE method 1: {}, with S0 = {}".format(payoff_type, LSMC_solver.y0[0],
                                                                                 LSMC_solver.x0[0]))

    if False:
        # LSMC Method 2
        FBSDE_american_put = american_option.FSBDEAmericanPut(cfg_am_put, method=2)
        LSMC_solver = LSMCLinear(FBSDE_american_put, cfg_lsmc_solver, basis_funcs_type='poly')
        LSMC_solver.solve()
        print("American {} option pricing by BSDE method 2: {}, with S0 = {}".format(payoff_type, LSMC_solver.y0[0],
                                                                                     LSMC_solver.x0[0]))

    # DeepBSDE Method 1
    if False:
        deep_solver = DeepBSDESolver(FBSDE, cfg_deep_solver)
        deep_solver.train()
        print("American {} option pricing by BSDE method 1: {}, with S0 = {}".format(payoff_type, deep_solver.y0[0],
                                                                                     cfg_deep_solver.x0[0]))


def main():

    payoff_type = 'vanilla'

    # Market Parameters
    r = 0.04
    sig = 0.6
    s0 = np.array([40])
    T = 1
    K = 40.
    lower_barrier = 30
    # lower_barrier = kwargs.get('lower_barrier', 20)
    # upper_barrier = kwargs.get('upper_barrier', 200)

    # Simulation parameters
    M = 2 ** 16
    dt = (1 / 252.)
    N = int(T / dt)
    d = 1
    d1 = 1
    d2 = 1

    # config for option and simulation
    config_am_put = ConfigOption(r=r, sig=sig, K=K, T=T, d1=d1, d2=d2, d=d)
    config_lsmc_solver = ConfigLSMC(M=M, N=N, dt=dt, x0=s0, seed=42,
                                    model_params={'fit_intercept': False}, reg_method=None)
    config_deep_solver = ConfigDeepBSDE(N=N, M=M, dt=dt, seed=42, x0=s0,
                                        y_init_range=[5, 10],
                                        n_hiddens=[10+d, 10+d],
                                        lr_values=[2e-3, 1e-3],
                                        lr_boundaries=[2000],
                                        n_iterations=4000,
                                        batch_size=256,
                                        valid_size=64,
                                        report_freq=100,
                                        dtype='float64',
                                        verbose=True)

    # BS
    if r == 0 and payoff_type == 'vanilla':
        BS_European_put = european_option.BS_EuroPut(S=s0[0], T=T, K=K, r=r, q=0, sig=sig)
        print("American/European vanilla put option by BS with r=0: {}, with S0 = {}".format(BS_European_put, s0[0]))

    # LSMC/Deep BSDE
    test_american_put(payoff_type, config_am_put, config_lsmc_solver, config_deep_solver, lower_barrier=lower_barrier)

    # PDE
    show_bs_pde_result(style='American', payoff_type=payoff_type, r=r, sig=sig, s0=s0, T=T, K=K, lower_barrier=lower_barrier)


if __name__ == '__main__':
    main()
