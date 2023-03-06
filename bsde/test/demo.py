import numpy as np
import matplotlib.pyplot as plt

import bsde.dynamics.european_option as eu
from bsde.solver.lsmc import generate_z_matrix

import scipy.sparse.linalg.dsolve as linsolve
from scipy import sparse


class BS_FDM_implicit:
    def __init__(self,
                 r,
                 sigma,
                 maturity,
                 Smin,
                 Smax,
                 Fl,
                 Fu,
                 payoff,
                 nt,
                 ns,
                 style):
        self.r = r
        self.sigma = sigma
        self.maturity = maturity

        self.Smin = Smin
        self.Smax = Smax
        self.Fl = Fl
        self.Fu = Fu

        self.nt = nt
        self.ns = ns

        self.dt = float(maturity) / nt
        self.dx = float(Smax - Smin) / (ns + 1)
        self.xs = Smin / self.dx

        self.u = np.empty((nt + 1, ns))
        self.u[0, :] = payoff
        self.style = style

        # Building Coefficient matrix:
        A = sparse.lil_matrix((self.ns, self.ns))

        for j in range(0, self.ns):
            xd = j + 1 + self.xs
            sx = self.sigma * xd
            sxsq = sx * sx

            dtmp1 = self.dt * sxsq
            dtmp2 = self.dt * self.r
            A[j, j] = 1.0 + dtmp1 + dtmp2

            dtmp1 = -0.5 * dtmp1
            dtmp2 = -0.5 * dtmp2 * xd
            if j > 0:
                A[j, j - 1] = dtmp1 - dtmp2
            if j < self.ns - 1:
                A[j, j + 1] = dtmp1 + dtmp2

        self.A = linsolve.splu(A)
        self.rhs = np.empty((self.ns,))

        # Building bc_coef:
        nxl = 1 + self.xs
        sxl = self.sigma * nxl
        nxu = self.ns + self.xs
        sxu = self.sigma * nxu

        self.blcoef = 0.5 * self.dt * (- sxl * sxl + self.r * nxl)
        self.bucoef = 0.5 * self.dt * (- sxu * sxu - self.r * nxu)

    def solve(self):
        for i in range(0, self.nt):
            self.rhs[:] = self.u[i, :]
            self.rhs[0] -= self.blcoef * self.Fl[i]
            self.rhs[self.ns - 1] -= self.bucoef * self.Fu[i]
            self.u[i + 1, :] = self.A.solve(self.rhs)
            if self.style == 'American':
                self.u[i + 1, :] = np.maximum(self.u[i + 1, :], self.u[0, :])

        return self.u


# PDE pricing
def show_bs_pde_result(style, payoff_type, r, sig, s0, T, K, **kwargs):

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

    if style == 'European':
        if payoff_type == 'vanilla':
            final_payoff = np.maximum(- K + S_s, 0)
            B_upper = np.exp(-r * t_s) * (S_max - K)
            B_lower = 0 * t_s
        elif payoff_type == 'barrier':
            upper_barrier = kwargs.get('upper_barrier', 1.5 * K)
            final_payoff = np.maximum(S_s - K, 0) * np.where((S_s <= upper_barrier), 1, 0)
            B_upper = 0 * t_s
            B_lower = 0 * t_s
    elif style == 'American':
        if payoff_type == 'vanilla':
            final_payoff = np.maximum(K - S_s, 0)
            B_upper = 0 * t_s
            B_lower = np.exp(-r * t_s) * K
        elif payoff_type == 'barrier':
            lower_barrier = kwargs.get('lower_barrier', 0.75 * K)
            final_payoff = np.maximum(K - S_s, 0) * np.where((S_s >= lower_barrier), 1, 0)
            B_upper = 0 * t_s
            B_lower = 0 * t_s

    BS_PDE_solver = BS_FDM_implicit(r, sig, T, S_min, S_max, B_lower, B_upper, final_payoff[1:-1], nt, ns, style=style)

    u_implicit = BS_PDE_solver.solve()
    n_row = len(u_implicit[:, 1])

    u = u_implicit[n_row - 1, :]
    s0_idx = int(2 * s0[0] - 1)
    print("{}-call-PDE: {}, with S0 = {}".format(payoff_type, u[s0_idx], S_s[s0_idx + 1]))


# Monte-Carlo
def mc_european(FBSDE, config_solver):
    s0 = config_solver.x0[0]
    dt = config_solver.dt
    M = config_solver.M
    N = config_solver.N
    T = FBSDE.T
    r = FBSDE.r
    dW = np.sqrt(config_solver.dt) * generate_z_matrix(n_paths=M, n_steps=N, d_bm=FBSDE.d, seed=config_solver.seed+1)

    S_T = FBSDE.draw(dW, s0, dt)[-1, 0, :]
    C0 = np.exp(-r * T) * sum(FBSDE.g(T, S_T)) / M

    return C0


def mc_european_batch(FBSDEs, configs_solver):
    MC_res = np.zeros(shape=(len(FBSDEs), len(configs_solver)))

    for (i, dynamics) in enumerate(FBSDEs):
        for (j, sim) in enumerate(configs_solver):
            p_cev = mc_european(dynamics, sim)
            MC_res[i, j] = p_cev
    return MC_res


# BS
def bs_european_call_batch(configs_option, s0):
    bs_res = np.zeros(len(configs_option))

    for (i, option) in enumerate(configs_option):
        BS_price = eu.BS_EuroCall(S=s0, T=option.T, K=option.K, r=option.r, q=0, sig=option.sig)
        bs_res[i] = BS_price

    return bs_res


def plot_convergence(solver_res, configs_option, configs_solver, benchmark_res=None):
    """
    plot the price of European Options for different
        1.M, number of path
    while other parameters stay the same.

    :param solver_res: Result from BSDE solvers, n_solver x n_options x n_sim_configs
    :param benchmark_res: Result from the benchmark method, either BS or MC
    :param configs_option: Options to be priced
    :param configs_solver: Solvers parameter
    :return:
    """
    fig = plt.figure(figsize=(8, 6), dpi=80)
    for (i, option) in enumerate(configs_option):

        # BSDE
        for res in solver_res:
            plt.plot(np.log2([cf.M for cf in configs_solver]), res[0, i, :])

        # Analytical
        if benchmark_res is not None:
            plt.axhline(benchmark_res[i], linestyle='-.')
            plt.legend(['LSMC-OLS', 'LSMC-ridge', 'Benchmark'])
        else:
            plt.legend(['LSMC-OLS', 'LSMC-ridge'])

        # set title, labels, etc.
        plt.title('American Put Option, sigma={}'.format(option.sig))
        plt.xlabel('Number of Path - log scale')
        plt.ylabel('Price')


def plot_price(solver_res, benchmark_res, s0s):
    fig = plt.figure(figsize=(8, 6), dpi=80)
    plt.plot(s0s, benchmark_res[0, :])
    plt.plot(s0s, solver_res[0][0, 0, :], linestyle='-.')
    plt.plot(s0s, solver_res[1][0, 0, :], linestyle='dashed')

    # set title, labels, etc.
    plt.xlabel('S0')
    plt.ylabel('Price')


def bs_pde_result(style, payoff_type, r, sig, s0, T, K, **kwargs):
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

    if style == 'European':
        if payoff_type == 'vanilla':
            final_payoff = np.maximum(- K + S_s, 0)
            B_upper = np.exp(-r * t_s) * (S_max - K)
            B_lower = 0 * t_s
        elif payoff_type == 'barrier':
            upper_barrier = kwargs.get('upper_barrier', 1.5 * K)
            final_payoff = np.maximum(S_s - K, 0) * np.where((S_s <= upper_barrier), 1, 0)
            B_upper = 0 * t_s
            B_lower = 0 * t_s
    elif style == 'American':
        if payoff_type == 'vanilla':
            final_payoff = np.maximum(K - S_s, 0)
            B_upper = 0 * t_s
            B_lower = np.exp(-r * t_s) * K
        elif payoff_type == 'barrier':
            lower_barrier = kwargs.get('lower_barrier', 0.75 * K)
            final_payoff = np.maximum(K - S_s, 0) * np.where((S_s >= lower_barrier), 1, 0)
            B_upper = 0 * t_s
            B_lower = 0 * t_s

    BS_PDE_solver = BS_FDM_implicit(r, sig, T, S_min, S_max, B_lower, B_upper, final_payoff[1:-1], nt, ns, style=style)

    u_implicit = BS_PDE_solver.solve()
    n_row = len(u_implicit[:, 1])

    u = u_implicit[n_row - 1, :]
    s0_idx = int(2 * s0[0] - 1)

    return u[s0_idx]


def pde_american_put_batch(payoff_type, configs_option, s0s):
    """
    Price for different spot, while fix other params
    """
    pde_res = np.zeros(shape=(len(configs_option), len(s0s)))

    for (i, option) in enumerate(configs_option):
        for (j, s0) in enumerate(s0s):
            BS_price = bs_pde_result('American', payoff_type, s0=[s0], T=option.T,
                                     K=option.K, r=option.r, sig=option.sig)
            pde_res[i, j] = BS_price

    return pde_res
