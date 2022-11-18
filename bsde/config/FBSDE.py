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


class config_HJB_liquidation1(config_FBSDE):
    def __init__(self, sig_s, k, eps, T, d, d1, d2):
        super(config_HJB_liquidation1, self).__init__(d, d1, d2)
        self.sig_s = sig_s
        self.k = k
        self.eps = eps
        self.T = T