import sys

sys.path.insert(0, '/Users/finn/Desktop/Capstone-BSDE/files/code')

import european_option
import LSMC
import numpy as np
import config as cf


def test_european_call():
    """
    Test the European call option price computed by FBSDE
    """
    # Market Parameters
    mu = 0.1
    r = 0.1
    sig = 0.8
    s0 = np.array([40])
    T = 1
    K = 30

    # Simulation parameters
    M = 2 ** 16
    dt = (1 / 252.)
    N = int(T / dt)
    d = 1
    d1 = 1
    d2 = 1

    # Config for option and simulation
    config = cf.config_option(r=r, sig=sig, K=K, T=T, d=d, d1=d1, d2=d2)
    config_sim = cf.config_simulation(N=N, M=M, dt=dt, seed=42, x0=s0)

    # Black-Scholes
    print('Black-Scholes: {}'.format(european_option.BS_EuroCall(S=s0, T=T, K=K, r=r, q=0, sig=sig)[0]))

    # LSMC
    BS_FBSDE = european_option.BS_FBSDE(config, method=2)

    # Linear Model
    LSMC_solver = LSMC.LSMC_linear(BS_FBSDE, config_sim, {'fit_intercept': False}, reg_method='ridge',
                                   basis_funcs_type='poly')
    LSMC_solver.solve()
    print('BSDE-linear: {}'.format(LSMC_solver.y0[0]))

    # Neural Net
    if False:
        LSMC_solver = LSMC.LSMC_neural_net(Y_sim, S_sim, dZ, s0, dt,
                                           model_params={'hidden_layer_sizes': (3,), 'max_iter': 5000,
                                                         'activation': 'logistic'},
                                           reg_method=None, basis_funcs_type='poly')
        LSMC_solver.solve()
        print('BSDE-Neural Net: {}'.format(LSMC_solver.y0[0]))

    # LSMC_solver = LSMC.LSMC_svm(Y_sim, S_sim, dZ, s0, dt, basis_funcs_type='trig')
    # print(LSMC_solver.alphas.shape)
    # print(LSMC_solver.betas.shape)


def main():
    test_european_call()


if __name__ == '__main__':
    main()
