# import sys
# sys.path.insert(0, '/Users/finn/Desktop/Capstone-BSDE/files/bsde')

import numpy as np
from bsde.config import ConfigOption, ConfigLSMC
from bsde.dynamics import european_option
from bsde.solver.lsmc import LSMCLinear, LSMCSVR, LSMCNeuralNet


def test_european_call():
    """
    Test the European call option price computed by dynamics
    """
    # Market Parameters
    mu = 0.1
    r = 0.1
    sig = 0.3
    s0 = np.array([40])
    T = 1
    K = 30

    # Simulation parameters
    M = 2 ** 18
    dt = (1 / 252.)
    N = int(T / dt)
    d = 1
    d1 = 1
    d2 = 1

    # Config and dynamics
    config_dynamics = ConfigOption(r=r, sig=sig, K=K, T=T, d=d, d1=d1, d2=d2)
    config_linear_solver = ConfigLSMC(N=N, M=M, dt=dt, seed=42, x0=s0, model_params={'fit_intercept': False}, reg_method=None)
    BS_FBSDE = european_option.BS_FBSDE(config_dynamics, method=2)

    # Black-Scholes
    print('Black-Scholes: {}'.format(european_option.BS_EuroCall(S=s0, T=T, K=K, r=r, q=0, sig=sig)[0]))

    # Solve the BSDE
    # Linear Model
    linear_solver = LSMCLinear(BS_FBSDE, config_linear_solver, basis_funcs_type='poly')
    linear_solver.solve()
    print('BSDE-lsmc-linear: {}'.format(linear_solver.y0[0]))

    # Neural Net -- takes too long to run
    if False:
        config_nn_solver = ConfigLSMC(N=N, M=M, dt=dt, seed=42, x0=s0,
                                      model_params={'hidden_layer_sizes': (3,), 'max_iter': 5000, 'activation': 'relu'})
        nn_solver = LSMCNeuralNet(BS_FBSDE, config_nn_solver, basis_funcs_type='poly')
        nn_solver.solve()
        print('BSDE-lsmc-Neural Net: {}'.format(nn_solver.y0[0]))

    # SVM
    if False:
        config_svr_solver = ConfigLSMC(N=N, M=M, dt=dt, seed=42, x0=s0, model_params={'kernel': 'rbf'})
        svr_solver = LSMCSVR(BS_FBSDE, config_svr_solver, basis_funcs_type='trig')
        svr_solver.solve()
        print('BSDE-lsmc-SVM: {}'.format(LSMC_solver.y0[0]))


def main():
    test_european_call()


if __name__ == '__main__':
    main()
