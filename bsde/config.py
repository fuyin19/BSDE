# PDE/FBSDE dynamics
class ConfigFBSDE(object):
    def __init__(self, d, d1, d2):
        self.d = d
        self.d1 = d1
        self.d2 = d2


class ConfigOption(ConfigFBSDE):
    def __init__(self, r, sig, K, T, d, d1, d2):
        super(ConfigOption, self).__init__(d, d1, d2)
        self.r = r
        self.sig = sig
        self.K = K
        self.T = T


class ConfigHJB1(ConfigFBSDE):
    def __init__(self, sig_s, k, eps, T, d, d1, d2):
        super(ConfigHJB1, self).__init__(d, d1, d2)
        self.sig_s = sig_s
        self.k = k
        self.eps = eps
        self.T = T


# Solver
class ConfigSolver(object):
    def __init__(self, x0, N, M, dt, seed=42):
        self.x0 = x0
        self.N = N
        self.M = M
        self.dt = dt
        self.seed = seed


class ConfigLSMC(ConfigSolver):
    def __init__(self, x0, N, M, dt, seed, model_params={}, basis_funcs=None, **kwargs):
        super(ConfigLSMC, self).__init__(x0=x0, N=N, M=M, dt=dt, seed=seed)
        self.model_params = model_params
        self.basis_funcs = basis_funcs
        self.kwargs = kwargs
