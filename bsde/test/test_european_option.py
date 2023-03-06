# import sys
# sys.path.insert(0, '/Users/finn/Desktop/Capstone-BSDE/files/bsde')

import numpy as np
import bsde.dynamics.european_option as eu

from bsde.test.demo import mc_european, show_bs_pde_result
from bsde.config import ConfigOption, ConfigLSMC, ConfigDeepBSDE
from bsde.solver.lsmc import LSMCLinear, LSMCSVR, LSMCNeuralNet, LSMCKernelRidge
from bsde.solver.deep_bsde import DeepBSDESolver


def test_bs_european_call(payoff_type, cfg_pde, cfg_linear_solver, cfg_kernel_solver, cfg_deep_bsde_solver):
    """
    Test the European call option price in BS model
    """
    bs_pde = eu.BSEuropeanCall(cfg_pde, payoff_type=payoff_type, exclude_spot=False, method=2)

    # Linear Model
    linear_solver = LSMCLinear(bs_pde, cfg_linear_solver, basis_funcs_type='poly')
    linear_solver.solve()
    print('{} call-BS-lsmc-linear: {}'.format(payoff_type, linear_solver.y0[0]))

    # Kernel-Ridge
    if False:
        kr_solver = LSMCKernelRidge(bs_pde, cfg_kernel_solver)
        kr_solver.solve()
        print('{} call-BS-lsmc-kr: {}'.format(payoff_type, kr_solver.y0[0]))

    # Deep BSDE
    if False:
        deep_solver = DeepBSDESolver(bs_pde, cfg_deep_bsde_solver)
        deep_solver.train()
        print(deep_solver.training_time)

    # Neural Net -- takes too long to run
    if False:
        config_nn_solver = ConfigLSMC(N=N, M=M, dt=dt, seed=42, x0=s0,
                                      model_params={'hidden_layer_sizes': (3,), 'max_iter': 5000, 'activation': 'relu'})
        nn_solver = LSMCNeuralNet(BS_FBSDE, config_nn_solver, basis_funcs_type='poly')
        nn_solver.solve()
        print('call-lsmc-Neural Net: {}'.format(nn_solver.y0[0]))

    # SVM
    if False:
        config_svr_solver = ConfigLSMC(N=N, M=M, dt=dt, seed=42, x0=s0, model_params={'kernel': 'rbf'})
        svr_solver = LSMCSVR(BS_FBSDE, config_svr_solver, basis_funcs_type='trig')
        svr_solver.solve()
        print('call-lsmc-SVM: {}'.format(LSMC_solver.y0[0]))
    print()


def test_cev_european_call(payoff_type, cfg_pde, cfg_lsmc_solver):
    """
    Test the European call option price in CEV model
    """
    # CEV
    print('CEV:')
    cev_pde = eu.BSEuropeanCallCEV(cfg_pde, payoff_type=payoff_type, beta=0.9)
    linear_solver = LSMCLinear(cev_pde, cfg_lsmc_solver, basis_funcs_type='poly')
    linear_solver.solve()
    print('{} call-CEV-lsmc-linear: {}'.format(payoff_type, linear_solver.y0[0]))

    p_cev = mc_european(cev_pde, cfg_lsmc_solver)
    print('call-CEV-MonteCarlo: {}'.format(p_cev))
    print()


def test_svi_european_call(payoff_type, cfg_pde, cfg_lsmc_solver):
    """
    Test the European call option price in SVI/local vol model
    """
    # SVI
    print('SVI:')
    params = {'a': 0.4, 'b': 0.04, 'rho': 0.1, 'm': 0.01, 'sigma': 30}

    svi_pde = eu.BSEuropeanCallSVI(cfg_pde, payoff_type=payoff_type, **params)
    linear_solver = LSMCLinear(svi_pde, cfg_lsmc_solver, basis_funcs_type='poly')
    linear_solver.solve()
    print('{} call-svi-lsmc-linear: {}'.format(payoff_type, linear_solver.y0[0]))

    p_cev = mc_european(svi_pde, cfg_lsmc_solver)
    print('{} call-svi-MonteCarlo: {}'.format(payoff_type, p_cev))


def main():
    payoff_type = 'vanilla'

    # Market Parameters
    # mu = 0.1
    r = 0.06
    sig = 0.5
    s0 = np.array([30])
    T = 1
    K = 30

    # Simulation parameters
    M = 2 ** 16
    dt = (1 / 252.)
    N = int(T / dt)
    d = 1
    d1 = 1
    d2 = 1

    # Config and dynamics
    cfg_pde = ConfigOption(r=r, sig=sig, K=K, T=T, d=d, d1=d1, d2=d2)
    cfg_lsmc_linear = ConfigLSMC(N=N, M=M, dt=dt, seed=42, x0=s0,
                                 model_params={'fit_intercept': False}, reg_method=None)
    cfg_lsmc_kr = ConfigLSMC(N=N, M=M, dt=dt, seed=42, x0=s0, model_params={'kernel': 'rbf'})

    cfg_deep_solver = ConfigDeepBSDE(N=N, M=M, dt=dt, seed=42, x0=s0,
                                     y_init_range=[1, 7],
                                     n_hiddens=[10+d, 10+d],
                                     lr_values=[5e-3, 9e-4, 5e-4],
                                     lr_boundaries=[2000, 3000],
                                     n_iterations=4000,
                                     batch_size=128,
                                     valid_size=64,
                                     report_freq=100,
                                     dtype='float64',
                                     verbose=True)

    # Black-Scholes
    print('Black-Scholes formula: {}'.format(eu.BS_EuroCall(S=s0, T=T, K=K, r=r, q=0, sig=sig)[0]))

    # PDE
    show_bs_pde_result(style='European', payoff_type=payoff_type, r=r, sig=sig, s0=s0, T=T, K=K)

    # run tests
    # test_bs_european_call(payoff_type, cfg_pde, cfg_lsmc_linear, cfg_lsmc_kr, cfg_deep_solver)
    test_cev_european_call(payoff_type, cfg_pde, cfg_lsmc_linear)
    test_svi_european_call(payoff_type, cfg_pde, cfg_lsmc_linear)


if __name__ == '__main__':
    main()
