import numpy as np
import matplotlib.pyplot as plt
import bsde.dynamics.european_option as eu


def mc_european_call(configs_option, configs_solver):

    MC_res = np.zeros(shape=(len(configs_option), len(configs_solver)))

    for (i, option) in enumerate(configs_option):
        for (j, sim) in enumerate(configs_solver):
            MC_res[i, j] = eu.MC_EuroCall(S_0=sim.x0[0], K=option.K, T=option.T,
                                          r=option.r, sigma=option.sig, M=sim.M, N=sim.N)
    return MC_res


def plot_european_call(solver_res, MC_res, configs_option, configs_solver):
    """
    plot the price of European Options for different
        1.sigmas, vol level
        2.M, number of path
    while other parameters stay the same.

    :param solver_res: Result from BSDE solvers, n_solver x n_options x n_sim_configs
    :param MC_res: Result from standard MC, n_options x n_sim_configs
    :param configs_option: Options to be priced
    :param configs_solver: Solvers parameter
    :return:
    """
    f, axes = plt.subplots(1, len(configs_option), figsize=(15, 4), dpi=80, sharex=True)
    for (i, option) in enumerate(configs_option):
        BS_price = eu.BS_EuroCall(S=configs_solver[0].x0[0], T=option.T, K=option.K, r=option.r, q=0, sig=option.sig)

        # BSDE
        for res in solver_res:
            axes[i].plot(np.log2([cf.M for cf in configs_solver]), res[0, i, :])

        # Monte-Carlo
        print(mc_res[i, :])
        axes[i].plot(np.log2([cf.M for cf in configs_solver]), MC_res[i, :])

        # BS
        axes[i].axhline(BS_price, linestyle='-.')

        # set labels, title, etc.
        axes[i].legend(['LSMC-OLS', 'LSMC-ridge', 'Monte-Carlo', 'BS'])
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
