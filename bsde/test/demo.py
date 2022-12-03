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


def plot_european_call(solver_res, MC_res, configs_option, configs_solver, analytical_res=None):
    """
    plot the price of European Options for different
        1.sigmas, vol level
        2.M, number of path
    while other parameters stay the same.

    :param solver_res: Result from BSDE solvers, n_solver x n_options x n_sim_configs
    :param MC_res: Result from standard MC, n_options x n_sim_configs
    :param configs_option: Options to be priced
    :param configs_solver: Solvers parameter
    :param analytical_res: Analytical Value
    :return: None
    """

    f, axes = plt.subplots(1, len(configs_option), figsize=(15, 4), dpi=80)
    for (i, option) in enumerate(configs_option):

        # BSDE
        for res in solver_res:
            axes[i].plot(np.log2([cf.M for cf in configs_solver]), res[0, i, :])

        # Monte-Carlo
        axes[i].plot(np.log2([cf.M for cf in configs_solver]), MC_res[i, :])

        # Analytical
        if analytical_res is not None:
            axes[i].axhline(analytical_res[i], linestyle='-.')
            axes[i].legend(['LSMC-OLS', 'LSMC-ridge', 'Monte-Carlo', 'Analytical'])
        else:
            axes[i].legend(['LSMC-OLS', 'LSMC-ridge', 'Monte-Carlo'])

        # set title, labels, etc.
        axes[i].set_title('sigma={}'.format(option.sig))
        axes[i].set_xlabel('Number of Path - log scale')
        axes[i].set_ylabel('Price')

    plt.tight_layout()
    plt.savefig('european_option_update.png', dpi=300, bbox_inches="tight")
    plt.show()


def plot_american_put(solver_res, PDE_res, configs_option, configs_solver):
    f, axes = plt.subplots(1, len(configs_option), figsize=(15, 4), dpi=80, sharex=True)
    for (i, option) in enumerate(configs_option):

        # BSDE
        for res in solver_res:
            axes[i].plot(np.log2([cf.M for cf in configs_solver]), res[0, i, :])

        # PDE
        axes[i].axhline(PDE_res[i], linestyle='-.')

        # set labels, title, etc.
        axes[i].legend(['LSMC-OLS', 'LSMC-ridge', 'PDE'])
        axes[i].set_title('sigma={}'.format(option.sig))
        axes[i].set_xlabel('Number of Path - log scale')
        axes[i].set_ylabel('Price')

    plt.tight_layout()
    plt.savefig('American_option_update.png', dpi=300, bbox_inches="tight")
    plt.show()


def PDE_european_call_cev(configs_option, configs_solver, beta=1):
    """
    Not Working currently
    """
    # result
    PDE_res = np.zeros(len(configs_option))

    # params
    s0 = configs_solver[0].x0[0]
    S_min = 0.0
    S_max = 400.
    nt = 5000
    ns = 799
    S_s = np.linspace(S_min, S_max, ns + 2)

    for (i, option) in enumerate(configs_option):
        t_s = np.linspace(0, option.T, nt + 1)
        final_payoff = np.maximum(S_s - option.K, 0)
        B_upper = np.exp(-option.r * t_s) * (S_max - option.K)
        B_lower = 0 * t_s

        BS_PDE_solver = BS_FDM_implicit(option.r,
                                        option.sig,
                                        option.T,
                                        S_min, S_max, B_lower, B_upper, final_payoff[1:-1], nt, ns,
                                        beta=beta)
        u = BS_PDE_solver.solve()[-1, :]

        s0_idx = int(2 * s0 - 1)

        PDE_res[i] = u[s0_idx]

    return PDE_res
