
class config_simulation(object):
    def __init__(self, N, M, dt, seed=42):
        self.N = N
        self.M = M
        self.dt = dt
        self.seed = seed


class config_FBSDE(object):
    def __init__(self, d, d1, d2):
        self.d = d
        self.d1 = d1
        self.d2 = d2


class config_option(config_FBSDE):
    def __init__(self, r, sig, K, T, d, d1, d2):
        super(config_option, self).__init__(d, d1, d2)
        self.r = r
        self.sig = sig
        self.K = K
        self.T = T
