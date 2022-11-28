# import sys
# sys.path.insert(0, '/Users/finn/Desktop/Capstone-BSDE/files/bsde')

import numpy as np
from bsde.test.demo import mc_european

from bsde.config import ConfigOption, ConfigLSMC
import bsde.dynamics.european_option as eu
from bsde.solver.lsmc import LSMCLinear, LSMCSVR, LSMCNeuralNet


def test_bs_european_call(cfg_pde, cfg_solver):
    """
    Test the European call option price computed by dynamics
    """
    bs_pde = eu.BS_FBSDE(cfg_pde, method=2)

    # Linear Model
    linear_solver = LSMCLinear(bs_pde, cfg_solver, basis_funcs_type='poly')
    linear_solver.solve()
    print('call-BS-lsmc-linear: {}'.format(linear_solver.y0[0]))

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


def test_cev_european_call(cfg_pde, cfg_solver):
    # CEV
    print('CEV:')
    cev_pde = eu.BS_CEV(cfg_pde, beta=0.9)
    linear_solver = LSMCLinear(cev_pde, cfg_solver, basis_funcs_type='poly')
    linear_solver.solve()
    print('call-CEV-lsmc-linear: {}'.format(linear_solver.y0[0]))

    p_cev = mc_european(cev_pde, cfg_solver)
    print('call-CEV-MonteCarlo: {}'.format(p_cev))
    print()


def test_svi_european_call(cfg_pde, cfg_solver):
    # SVI
    print('SVI:')
    params = {'a': 0.4, 'b': 0.04, 'rho': 0.1, 'm': 0.01, 'sigma': 30}

    svi_pde = eu.BS_SVI(cfg_pde, **params)
    linear_solver = LSMCLinear(svi_pde, cfg_solver, basis_funcs_type='poly')
    linear_solver.solve()
    print('call-svi-lsmc-linear: {}'.format(linear_solver.y0[0]))

    p_cev = mc_european(svi_pde, cfg_solver)
    print('call-svi-MonteCarlo: {}'.format(p_cev))


def main():
    # Market Parameters
    # mu = 0.1
    r = 0.03
    sig = 0.8
    s0 = np.array([40])
    T = 1
    K = 30

    # Simulation parameters
    M = 2 ** 17
    dt = (1 / 252.)
    N = int(T / dt)
    d = 1
    d1 = 1
    d2 = 1

    # Config and dynamics
    cfg_pde = ConfigOption(r=r, sig=sig, K=K, T=T, d=d, d1=d1, d2=d2)
    cfg_lsmc_linear = ConfigLSMC(N=N, M=M, dt=dt, seed=42, x0=s0,
                                 model_params={'fit_intercept': False}, reg_method=None)

    # Black-Scholes
    print('Black-Scholes formula: {}'.format(eu.BS_EuroCall(S=s0, T=T, K=K, r=r, q=0, sig=sig)[0]))

    # run tests
    test_bs_european_call(cfg_pde, cfg_lsmc_linear)
    test_cev_european_call(cfg_pde, cfg_lsmc_linear)
    test_svi_european_call(cfg_pde, cfg_lsmc_linear)


if __name__ == '__main__':
    main()
