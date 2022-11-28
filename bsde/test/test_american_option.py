# import sys
# sys.path.insert(0, '/Users/finn/Desktop/Capstone-BSDE/files/bsde')

from bsde.dynamics import american_option, european_option
from bsde.config import ConfigOption, ConfigLSMC
from bsde.solver.lsmc import LSMCLinear

import numpy as np


def test_american_put(payoff_type="vanilla", **kwargs):
    """
    Test the american put option price computed by dynamics
    """
    # Market Parameters
    r = 0.6
    sig = 0.4
    s0 = np.array([40])
    T = 1.
    K = 40.
    lower_barrier = kwargs.get('lower_barrier', 20)
    upper_barrier = kwargs.get('upper_barrier', 200)

    # Simulation parameters
    M = 2 ** 16
    dt = (1 / 252.)
    N = int(T / dt)
    d = 1
    d1 = 1
    d2 = 1

    # configg for option and simulation
    config_dynamics = ConfigOption(r=r, sig=sig, K=K, T=T, d1=d1, d2=d2, d=d)
    config_solver = ConfigLSMC(M=M, N=N, dt=dt, x0=s0, seed=42,
                               model_params={'fit_intercept': False}, reg_method='ridge')

    # BSDE Method 1
    FBSDE_american_put = american_option.BS_american_FBSDE(config_dynamics, method=1)
    LSMC_solver = LSMCLinear(FBSDE_american_put, config_solver, basis_funcs_type='poly')
    LSMC_solver.solve()
    print("American {} option pricing by BSDE method 1: {}, with S0 = {}".format(payoff_type, LSMC_solver.y0[0], s0[0]))

    # BSDE Method 2
    FBSDE_american_put = american_option.BS_american_FBSDE(config_dynamics, method=2)
    LSMC_solver = LSMCLinear(FBSDE_american_put, config_solver, basis_funcs_type='poly')
    LSMC_solver.solve()
    print("American {} option pricing by BSDE method 1: {}, with S0 = {}".format(payoff_type, LSMC_solver.y0[0], s0[0]))

    # PDE - variational equation
    S_min = 0.0
    S_max = 700

    nt = 5000
    ns = 1399
    S_s = np.linspace(S_min, S_max, ns + 2)
    t_s = np.linspace(0, T, nt + 1)
    final_payoff = None
    B_upper = None
    B_lower = None

    if payoff_type == 'vanilla':
        final_payoff = np.maximum(K - S_s, 0)
        B_upper = 0 * t_s
        B_lower = np.exp(-r * t_s) * K
    elif payoff_type == 'barrier':
        final_payoff = np.maximum(K - S_s, 0) * np.where((S_s >= lower_barrier) & (S_s <= upper_barrier), 1, 0)
        B_upper = 0 * t_s
        B_lower = 0 * t_s

    BS_PDE_solver = american_option.BS_FDM_implicit(r, sig, T, S_min, S_max, B_lower, B_upper,
                                                    final_payoff[1:-1], nt, ns)

    u_implicit = BS_PDE_solver.solve()
    n_row = len(u_implicit[:, 1])

    u = u_implicit[n_row - 1, :]
    s0_idx = int(2 * s0[0] - 1)
    print("American {} option pricing by PDE: {}, with S0 = {}".format(payoff_type, u[s0_idx], S_s[s0_idx + 1]))

    if r == 0 and payoff_type == 'vanilla':
        BS_European_put = european_option.BS_EuroPut(S=s0[0], T=T, K=K, r=r, q=0, sig=sig)
        print("American/European vanilla put option by BS with r=0: {}, with S0 = {}".format(BS_European_put, s0[0]))


def main():
    print()
    test_american_put(payoff_type='vanilla')


if __name__ == '__main__':
    main()
