import european_option
import american_option
import Liquidation1
import LSMC
import numpy as np
import LSMC_engine


def test_european_call():
    """
    test the European call option price computed by FBSDE
    """
    # Market Parameters
    mu = 0.1
    r = 0.1
    sig = 0.3
    s0 = np.array([40])
    T = 1
    K = 40

    # Simulation parameters
    M = 2 ** 14
    dt = 1 / 252.
    N = int(T/dt)
    d = 1
    d1 = 1
    d2 = 1

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
    print(european_option.BS_EuroCall(S=s0, T=T, K=K, r=r, q=0, sig=sig)[0])

    # print(LSMC_solver.alphas.shape)
    # print(LSMC_solver.betas.shape)


def test_american_put():
    """
    test the american put option price computed by FBSDE
    """
    # Market Parameters
    r = 0.
    sig = 0.9
    s0 = np.array([30])
    T = 1
    K = 50

    # Simulation parameters
    M = 2 ** 14
    dt = 1 / 252.
    N = int(T/dt)
    d = 1
    d1 = 1
    d2 = 1
    dZ = LSMC.generate_z_matrix(n_paths=M, n_steps=N, d_bm=d)

    # BSDE - LSMC
    S_sim = american_option.S_t(mu=r, sig=sig, d=d, d1=d1)
    Y_sim = american_option.Y_t(d2=d2, mu=r, sig=sig, K=K)

    LSMC_solver = LSMC.LSMC_linear(Y_sim, S_sim, dZ, s0, dt, reg_method=None, basis_funcs_type='poly')
    LSMC_solver.solve()
    print("American vanilla option pricing by BSDE: {}, with S0 = {}".format(LSMC_solver.y0[0], s0[0]))

    # PDE - variational equation
    S_min = 0
    S_max = s0[0]*2

    nt = N
    ns = 2*S_max-1
    S_s = np.linspace(S_min, S_max, ns+2)
    t_s = np.linspace(0, T, nt+1)

    final_payoff = np.maximum(K - S_s, 0)
    B_upper = 0*t_s
    B_lower = np.exp(-r*t_s) * K

    BS_PDE_solver = american_option.BS_FDM_implicit(r, sig, T, S_min, S_max, B_lower, B_upper,
                                                    final_payoff[1:-1], nt, ns)
    u_implicit = BS_PDE_solver.solve()

    u = u_implicit[-1, :]
    s0_idx = 2*s0[0]
    print("American vanilla option pricing by PDE: {}, with S0 = {}".format(u[s0_idx], S_s[s0_idx]))

    BS_European_put = european_option.BS_EuroPut(S=s0[0], T=T, K=K, r=r, q=0, sig=sig)
    print("American vanilla put option by BS with r=0: {}, with S0 = {}".format(BS_European_put, s0[0]))


def test_liquidation1():
    """
    test the PDE solver using the example in section 6.3
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


def test_engine():
    # Market Parameters
    s0 = np.array([40])
    T = 1

    # Simulation parameters
    M = 2 ** 14
    dt = 1 / 252.
    d = 1
    d1 = 1
    d2 = 1

    LSMC_engine_european = LSMC_engine.LSMC_engine(d1=d1, d2=d2, d=d, M=M, dt=dt, T=T, x0=s0, task='pricing', r=[0.1, 0.2], sig=[0.3, 0.5], K=[40, 50])
    LSMC_engine_european.run()



def main():
    # test_engine()
    test_american_put()
    # test_european_call()
    # test_liquidation1()


if __name__ == '__main__':
    main()

