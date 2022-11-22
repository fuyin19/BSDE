# import sys
# sys.path.insert(0, '/Users/finn/Desktop/Capstone-BSDE/files/bsde')

import numpy as np
from bsde.config.FBSDE import config_option
from bsde.config.simulation import config_simulation
from bsde.config.LSMC import config_linear, config_svm, config_NN

from bsde.dynamics import european_option
from bsde.LSMC import LSMC


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
    M = 2 ** 14
    dt = (1 / 252.)
    N = int(T / dt)
    d = 1
    d1 = 1
    d2 = 1

    # Config and dynamics
    config_dynamics = config_option(r=r, sig=sig, K=K, T=T, d=d, d1=d1, d2=d2)
    config_sim = config_simulation(N=N, M=M, dt=dt, seed=42, x0=s0)
    config_model = config_linear(model_params={'fit_intercept': False}, reg_method=None)
    BS_FBSDE = european_option.BS_FBSDE(config_dynamics, method=2)

    # Black-Scholes
    print('Black-Scholes: {}'.format(european_option.BS_EuroCall(S=s0, T=T, K=K, r=r, q=0, sig=sig)[0]))

    # LSMC
    # Linear Model
    LSMC_solver = LSMC.LSMC_linear(BS_FBSDE, config_sim, config_model, basis_funcs_type='poly')
    LSMC_solver.solve()
    print('BSDE-LSMC-linear: {}'.format(LSMC_solver.y0[0]))

    # Neural Net -- takes too long to run
    config_model = config_NN(model_params={'hidden_layer_sizes': (3,), 'max_iter': 5000, 'activation': 'relu'}, )
    if False:
        LSMC_solver = LSMC.LSMC_neural_net(BS_FBSDE, config_sim, config_model, basis_funcs_type='poly')
        LSMC_solver.solve()
        print('BSDE-LSMC-Neural Net: {}'.format(LSMC_solver.y0[0]))

    # SVM
    if False:
        config_model = config_svm(model_params={'kernel': 'rbf'})
        LSMC_solver = LSMC.LSMC_svm(BS_FBSDE, config_sim, config_model, basis_funcs_type='trig')
        LSMC_solver.solve()
        print('BSDE-LSMC-SVM: {}'.format(LSMC_solver.y0[0]))


def main():
    test_european_call()


if __name__ == '__main__':
    main()
