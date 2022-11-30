import numpy as np
import scipy.sparse.linalg.dsolve as linsolve
import tensorflow as tf
from bsde.dynamics.fbsde import FBSDE
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
                 ns):
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

        return self.u


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
class BS_FBSDE(FBSDE):
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
        self.method = kwargs.get('method', 1)

    def mu_t(self, t, s):
        return self.mu * s

    def sig_t(self, t, s):
        return np.array([self.sig * s])

    def f(self, t, x, y, z):
        if self.method == 1:
            return -1 * ((self.mu - self.r) / self.sig * z[0, 0] + self.r * y)
        elif self.method == 2:
            return -self.r * y

    def g(self, T, x, use_tensor=False):
        if use_tensor:
            return tf.math.maximum(-self.K + x, 0)
        else:
            return np.maximum(-self.K + x, 0)


class BS_CEV(BS_FBSDE):
    def __init__(self, config, exclude_spot=False, **kwargs):
        super(BS_CEV, self).__init__(config, exclude_spot, **kwargs)
        self.frac = kwargs.get('beta', 0.9)

    def sig_t(self, t, s):
        return np.array([self.sig * s**self.frac])


class BS_SVI(BS_FBSDE):
    def __init__(self, config, exclude_spot=False, **kwargs):
        super(BS_SVI, self).__init__(config, exclude_spot, **kwargs)
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
