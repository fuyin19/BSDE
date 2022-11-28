import numpy as np
import matplotlib.pyplot as plt
import bsde.dynamics.european_option as eu
from bsde.solver.lsmc import generate_z_matrix


# Monte-Carlo
def mc_european(FBSDE, config_solver):
    s0 = config_solver.x0[0]
    dt = config_solver.dt
    M = config_solver.M
    N = config_solver.N
    T = FBSDE.T
    K = FBSDE.K
    r = FBSDE.r
    dW = np.sqrt(config_solver.dt) * generate_z_matrix(n_paths=M, n_steps=N, d_bm=FBSDE.d, seed=config_solver.seed+1)

    S_T = FBSDE.draw(dW, s0, dt)[-1, 0, :]
    C0 = np.exp(-r * T) * sum(FBSDE.g(T, S_T)) / M

    return C0


def mc_european_batch(FBSDEs, configs_solver):
    MC_res = np.zeros(shape=(len(configs_option), len(configs_solver)))

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
    :return:
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
        B_upper = np.exp(-r * t_s) * (S_max - option.K)
        B_lower = 0 * t_s

        BS_PDE_solver = eu.BS_FDM_implicit(option.r,
                                           option.sig,
                                           option.T,
                                           S_min, S_max, B_lower, B_upper, final_payoff[1:-1], nt, ns,
                                           beta=beta)
        u = BS_PDE_solver.solve()[-1, :]

        s0_idx = int(2 * s0 - 1)

        PDE_res[i] = u[s0_idx]

    return PDE_res
