import sys
sys.path.insert(0, '/Users/finn/Desktop/Capstone-BSDE/files/code')

import american_option
import european_option
import LSMC
import numpy as np


def test_american_put():
    """
    Test the american put option price computed by FBSDE
    """
    # Market Parameters
    r = 0.
    sig = 0.9
    s0 = np.array([30])
    T = 1
    K = 50

    # Simulation parameters
    M = 2 ** 16
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
    S_min = 0.0
    S_max = 200

    nt = 4555
    ns = 199
    S_s = np.linspace(S_min, S_max, ns+2)
    t_s = np.linspace(0, T, nt+1)

    final_payoff = np.maximum(K - S_s, 0)
    B_upper = 0*t_s
    B_lower = np.exp(-r*t_s) * K

    BS_PDE_solver = american_option.BS_FDM_implicit(r, sig, T, S_min, S_max, B_lower, B_upper,
                                                    final_payoff[1:-1], nt, ns)

    u_implicit = BS_PDE_solver.solve()
    nrow = len(u_implicit[:, 1])

    u = u_implicit[nrow-1, :]
    s0_idx = int(s0[0]-1)
    print("American vanilla option pricing by PDE: {}, with S0 = {}".format(u[s0_idx], S_s[s0_idx+1]))

    if r == 0:
        BS_European_put = european_option.BS_EuroPut(S=s0[0], T=T, K=K, r=r, q=0, sig=sig)
        print("American/European vanilla put option by BS with r=0: {}, with S0 = {}".format(BS_European_put, s0[0]))


def main():
    print()
    test_american_put()


if __name__ == '__main__':
    main()

