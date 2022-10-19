import european_option
import Liquidation1
import LSMC
import numpy as np


def test_european_call():
    """
    Test the European call option price computed by FBSDE
    """
    # Market Parameters
    mu = 0.1
    r = 0.03
    sig = 0.3
    s0 = np.array([40])
    T = 1
    K = 30

    # Simulation parameters
    M = 2 ** 14
    N = 100
    d = 1
    d1 = 1
    d2 = 1
    dt = T / N
    dZ = LSMC.generate_z_matrix(n_paths=M, n_steps=N, d_bm=d)

    # LSMC
    basis_funcs = [lambda x, coef=i: x ** coef for i in range(3)]

    S_sim = european_option.S_t(mu, sig, d, d1)
    Y_sim = european_option.Y_t(mu, r, sig, d2, K)

    LSMC_solver = LSMC.LSMC_linear(Y_sim, S_sim, dZ, s0, dt, reg_method=None, basis_funcs_type='poly')
    # LSMC_solver = LSMC.LSMC_svm(Y_sim, S_sim, dZ, s0, dt, basis_funcs_type='trig')

    LSMC_solver.solve()

    # Pricing with LSMC of FBSDE
    print(LSMC_solver.y0[0])
    # print(MC_EuroCall(S_0=s0, K=K, T=T, r=r, sigma=sig, M=N, N=M))
    print(european_option.BS_EuroCall(S=s0, T=T, K=K, r=r, q=0, sig=sig)[0])

    # print(LSMC_solver.alphas.shape)
    # print(LSMC_solver.betas.shape)


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
    test_european_call()
    # test_liquidation1()


if __name__ == '__main__':
    main()

