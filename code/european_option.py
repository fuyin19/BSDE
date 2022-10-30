from FBSDE import *


def MC_EuroCall(S_0, K, T, r, sigma, N, M, seed=42):
    """
    Monte Carlo Simulation for pricing a European Call option.

    Inputs:
        S_0: initial stock price
        K: strike price
        T: time to  maturity in years
        r: riskless interest rate
        sigma: volatility of the stock price
        N: the number of iterations to run
        M: the number of time-steps

    Output: Prints the European Call Value

    RRL March 23, 2021: adapted from Hilpisch "Python for Finance"
    """
    np.random.seed(seed)
    dt = T / M

    # Simulating N paths with M time steps
    S = S_0 * np.exp(np.cumsum((r - 0.5 * sigma ** 2) * dt
                               + sigma * np.sqrt(dt)
                               * np.random.standard_normal((M + 1, N)), axis=0))

    # Calculating the Monte Carlo estimator
    C0 = np.exp(-r * T) * sum(np.maximum(S[-1] - K, 0)) / N

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


# Implement the FBSDE
class S_t(FSDE):
    """
    X_t for European vanilla option
    """
    def __init__(self, mu, sig, d, d1, exclude_spot=False):
        super().__init__(d, d1, exclude_spot)
        self.mu = mu
        self.sig = sig

    def mu_t(self, t, s):
        return self.mu * s

    def sig_t(self, t, s):
        return np.array([self.sig * s])


class Y_t(BSDE):
    """
    Y_t for European vanilla option
    """
    def __init__(self, mu, r, sig, d2, K):
        super().__init__(d2)
        self.mu = mu
        self.r = r
        self.sig = sig
        self.K = K

    def f(self, t, x, y, z):
        return -1 * ((self.mu - self.r) / self.sig * z[0, 0] + self.r * y)

    def g(self, T, x, level=0.01):
        return np.maximum(-self.K + x, 0)
