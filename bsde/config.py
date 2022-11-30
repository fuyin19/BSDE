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


class ConfigDeepBSDE(ConfigSolver):
    def __init__(self, x0, N, dt, seed, n_hiddens, y_init_range, lr_values,
                 lr_boundaries, n_iterations, batch_size, valid_size, report_freq, dtype, verbose, **kwargs):
        super(ConfigDeepBSDE, self).__init__(x0=x0, N=N, M=(n_iterations+1)*batch_size, dt=dt, seed=seed)
        self.n_hiddens = n_hiddens
        self.y_init_range = y_init_range
        self.lr_values = lr_values
        self.lr_boundaries = lr_boundaries
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.valid_size = valid_size
        self.report_freq = report_freq
        self.dtype = dtype
        self.verbose = verbose
        self.kwargs = kwargs
