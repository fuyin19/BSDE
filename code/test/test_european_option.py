import sys
sys.path.insert(0, '/Users/finn/Desktop/Capstone-BSDE/files/code')

import european_option
import LSMC
import numpy as np


def test_european_call():
    """
    Test the European call option price computed by FBSDE
    """
    # Market Parameters
    mu = 0.06
    r = 0.06
    sig = 0.25
    s0 = np.array([40])
    T = 1
    K = 40

    # Simulation parameters
    M = 2 ** 16
    dt = (1 / 252.)
    N = int(T/dt)
    d = 1
    d1 = 1
    d2 = 1

    dZ = LSMC.generate_z_matrix(n_paths=M, n_steps=N, d_bm=d)

    # LSMC
    basis_funcs = [lambda x, coef=i: x ** coef for i in range(3)]

    S_sim = european_option.S_t(mu, sig, d, d1)
    Y_sim = european_option.Y_t(mu, r, sig, d2, K, method=2)

    LSMC_solver = LSMC.LSMC_linear(Y_sim, S_sim, dZ, s0, dt, reg_method=None, basis_funcs_type='poly')
    # LSMC_solver = LSMC.LSMC_svm(Y_sim, S_sim, dZ, s0, dt, basis_funcs_type='trig')

    LSMC_solver.solve()

    # Pricing with LSMC of FBSDE
    print(LSMC_solver.y0[0])
    print(european_option.BS_EuroCall(S=s0, T=T, K=K, r=r, q=0, sig=sig)[0])

    # print(LSMC_solver.alphas.shape)
    # print(LSMC_solver.betas.shape)


def main():
    test_european_call()


if __name__ == '__main__':
    main()

