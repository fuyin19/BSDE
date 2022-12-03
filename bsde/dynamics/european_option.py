import numpy as np
import tensorflow as tf
from bsde.dynamics.fbsde import FBSDE


def MC_EuroCall(S_0, K, T, r, sigma, M, N, seed=42):
    """
    Monte Carlo Simulation for pricing a European Call option.

    Inputs:
        S_0: initial stock price
        K: strike price
        T: time to  maturity in years
        r: riskless interest rate
        sigma: volatility of the stock price
        M: the number of iterations to run
        N: the number of time-steps

    Output: Prints the European Call Value

    RRL March 23, 2021: adapted from Hilpisch "Python for Finance"
    """
    np.random.seed(seed)
    dt = T / N

    # Simulating M paths with N time steps
    S = S_0 * np.exp(np.cumsum((r - 0.5 * sigma ** 2) * dt
                               + sigma * np.sqrt(dt)
                               * np.random.standard_normal((N + 1, M)), axis=0))

    # Calculating the Monte Carlo estimator
    C0 = np.exp(-r * T) * sum(np.maximum(S[-1] - K, 0)) / M

    return C0


def BS_EuroCall(S, T, K, r, q, sig):
    """
    Unsafe - arguments needs to be valid
    BS call option pricing
    """
    import math
    from scipy.stats import norm
    FT = S * np.exp((r - q) * T)
    total_vol = sig * math.sqrt(T)

    d_1 = 1 / total_vol * np.log(FT / K) + 0.5*total_vol
    d_2 = d_1 - total_vol

    return (FT * norm.cdf(d_1) - K * norm.cdf(d_2)) * np.exp(-r * T)


def BS_EuroPut(S, T, K, r, q, sig):
    """
    BS European put option price by put-call parity
    """

    discount = np.exp(-r * T)
    call = BS_EuroCall(S=S, T=T, K=K, r=r, q=q, sig=sig)

    return call - S + discount * K


# Implement the dynamics
class BSEuropeanCall(FBSDE):
    """
    Forward Backward dynamic representation of Black-Scholes PDE for European vanilla option
    """
    def __init__(self, config, exclude_spot=False, **kwargs):
        super().__init__(config, exclude_spot)
        self.mu = config.r
        self.sig = config.sig
        self.r = config.r
        self.K = config.K
        self.T = config.T

        self.method = kwargs.get('method', 2)
        self.payoff_type = kwargs.get('payoff_type', 'vanilla')
        if self.payoff_type == 'barrier':
            self.upper_barrier = kwargs.get('upper_barrier', 1.5*self.K)

    def mu_t(self, t, s):
        return self.mu * s

    def sig_t(self, t, s):
        return np.array([self.sig * s])

    def f(self, t, x, y, z, use_tensor=False):
        if self.method == 1:
            return -1 * ((self.mu - self.r) / self.sig * z[0, 0] + self.r * y)
        elif self.method == 2:
            return -self.r * y

    def g(self, T, x, use_tensor=False):
        maximum = tf.math.maximum if use_tensor else np.maximum
        if self.payoff_type == 'vanilla':
            return maximum(-self.K + x, 0)
        elif self.payoff_type == 'barrier':
            where = tf.where if use_tensor else np.where
            return maximum(-self.K + x, 0) * where(x < self.upper_barrier, 1, 0)


class BSEuropeanCallCEV(BSEuropeanCall):
    def __init__(self, config, exclude_spot=False, **kwargs):
        super(BSEuropeanCallCEV, self).__init__(config, exclude_spot, **kwargs)
        self.beta = kwargs.get('beta', 0.9)

    def sig_t(self, t, s):
        return np.array([self.sig * s**self.beta])


class BSEuropeanCallSVI(BSEuropeanCall):
    def __init__(self, config, exclude_spot=False, **kwargs):
        super(BSEuropeanCallSVI, self).__init__(config, exclude_spot, **kwargs)
        self.a = kwargs.get('a', -4)
        self.b = kwargs.get('b', 0.8)
        self.rho = kwargs.get('rho', 0.15)
        self.m = kwargs.get('m', 0.9)
        self.sigma = kwargs.get('sigma', 5)

    def raw_svi(self, t, s):
        FT = self.K * np.exp(self.r * (self.T-t))
        k = np.log(s/FT)

        return self.a + self.b * (self.rho * (k - self.m) + np.sqrt((k - self.m) ** 2 + self.sig ** 2))

    def sig_t(self, t, s):
        implied_vol = self.raw_svi(t, s)
        return np.array([implied_vol * s])
