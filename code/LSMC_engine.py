import numpy as np
import LSMC
import european_option
import american_option
import Liquidation1


class LSMC_engine(object):
    def __init__(self, d1, d2, d, task, M, dt, T, x0, **kwargs):
        self.task = task
        self.d1 = d1
        self.d2 = d2
        self.d = d

        self.T = T
        self.M = M
        self.dt = dt
        self.N = int(self.T/dt)
        self.dZ = LSMC.generate_z_matrix(n_paths=self.M, n_steps=self.N, d_bm=self.d)
        self.x0 = x0

        if task == 'pricing':
            self.option_type = kwargs.get('option_type', 'European')
            self.r_s = np.array(kwargs.get('r'))  # N_r
            self.sig_s = np.array(kwargs.get('sig'))  # N_sig
            self.K_s = np.array(kwargs.get('K'))  # N_K

            res_shape = (len(self.r_s), len(self.sig_s), len(self.K_s))
            self.prices = np.zeros(shape=res_shape)

    def allocate_idx(self):
        if self.task == 'pricing':
            idx_r, idx_sig, idx_K = np.arange(len(self.r_s)), np.arange(len(self.sig_s)), np.arange(len(self.K_s))
            idx_rr, idx_ss, idx_KK = (i.flatten() for i in np.meshgrid(idx_r, idx_sig, idx_K))
            return idx_rr, idx_ss, idx_KK

    def run(self):
        if self.task == 'pricing':
            idx_rr, idx_ss, idx_KK = self.allocate_idx()
            for i in range(len(self.prices.flatten())):
                r = self.r_s[idx_rr[i]]
                sig = self.sig_s[idx_ss[i]]
                K = self.K_s[idx_KK[i]]
                S_sim = european_option.S_t(mu=r, sig=sig, d=self.d, d1=self.d1)

                if self.option_type == 'European':
                    Y_sim = european_option.Y_t(mu=r, r=r, sig=sig, d2=self.d2, K=K)
                elif self.option_type == 'American':
                    Y_sim = american_option.Y_t(mu=r, sig=sig, d2=self.d2, K=K, T=self.T)

                LSMC_solver = LSMC.LSMC_linear(Y_sim, S_sim, self.dZ, self.x0, self.dt,
                                               reg_method=None, basis_funcs_type='poly')

                LSMC_solver.solve()
                self.prices[idx_rr[i], idx_ss[i], idx_KK[i]] = LSMC_solver.y0[0]
                # print('r={}, sig={}, K={}: {}'.format(r, sig, K, LSMC_solver.y0[0]))





