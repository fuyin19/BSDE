# Implement the dynamics

import numpy as np
import tensorflow as tf
import scipy.sparse.linalg.dsolve as linsolve
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

            # since it's an american option
            self.u[i + 1, :] = np.maximum(self.u[i + 1, :], self.u[0, :])
            # print(self.u[i + 1, :])
            # print(self.u[0, :])
            # print(np.maximum(self.u[i + 1, :], self.u[0, :]))

        return self.u


class FSBDEAmericanPut(FBSDE):
    """
    FBSDE for American vanilla option
    """
    def __init__(self, config, exclude_spot=False, **kwargs):
        super().__init__(config, exclude_spot)
        self.r = config.r
        self.mu = config.r
        self.sig = config.sig
        self.K = config.K
        self.T = config.T

        self.method = kwargs.get('method', 1)
        self.level = kwargs.get('level', 0.001)

    def mu_t(self, t, s):
        return self.mu * s

    def sig_t(self, t, s):
        return np.array([self.sig * s])

    def f(self, t, x, y, z, use_tensor=False):
        eps = self.level * self.K
        a = self.K - eps
        b = self.K + eps

        where = np.where if not use_tensor else tf.where

        if self.method == 1:
            # Check Continuation Region
            g_val = self.g(t, x, use_tensor)  # d2 x M
            indicator_ex = where(y - g_val <= 0, 1, 0)  # d2 x M

            # Compute Lg
            partial_x = where(x < a, -1, 0) + where((x >= a) & (x <= b), 1 / (2 * eps) * (x - b), 0)  # d2 x M
            partial_xx = where((x >= a) & (x <= b), 1 / (2 * eps), 0)  # d2 x M
            Lg = self.mu * x * partial_x + 0.5 * (self.sig ** 2) * (x ** 2) * partial_xx  # d2 x M

            # Compute (Lg - rg)^-
            val = Lg - self.mu * g_val
            val_minus = -1 * where(val <= 0, val, 0)  # d2 x M

            # Compute f = -ru + (Lg - rg)^- * I
            return -self.mu * y + val_minus * indicator_ex

        if self.method == 2:
            # Check Continuation Region
            g_val = np.maximum(self.K - x, 0)
            indicator_ex = np.where(y * np.exp(self.mu * t) - g_val <= 0, 1, 0)  # d2 x M

            # Compute Lg
            partial_x = np.where(x < a, -1, 0) + np.where((x >= a) & (x <= b), 1 / (2 * eps) * (x - b), 0)  # d2 x M
            partial_xx = np.where((x >= a) & (x <= b), 1 / (2 * eps), 0)  # d2 x M
            Lg = self.mu * x * partial_x + 0.5 * (self.sig ** 2) * (x ** 2) * partial_xx  # d2 x M

            # Compute c = -Lg + rg
            c = -Lg + self.mu * g_val

            #  Compute q = disc * c * indicator_ex
            return np.exp(-self.mu * t) * c * indicator_ex

    def g(self, T, x, use_tensor=False):

        maximum = tf.math.maximum if use_tensor else np.maximum
        exp = tf.math.exp if use_tensor else np.exp

        if self.method == 1:
            return maximum(self.K - x, 0)
        if self.method == 2:
            return maximum(self.K - x, 0) * exp(-self.mu * T)


class FBSDEAmericanPutBarrier(FSBDEAmericanPut):
    """
    FBSDE for American barrier option (not using approximation)
    """
    def __init__(self, config, exclude_spot=False, **kwargs):
        super(FBSDEAmericanPutBarrier, self).__init__(config, exclude_spot)
        self.lower_barrier = kwargs.get('lower_barrier', 0.75*self.K)

    def f(self, t, x, y, z, use_tensor=False):

        where = np.where if not use_tensor else tf.where

        # Check Continuation Region
        g_val = self.g(t, x, use_tensor)  # d2 x M
        indicator_ex = where(y - g_val <= 0, 1, 0)  # d2 x M

        # Compute Lg
        partial_x = where((x > self.lower_barrier) & (x < self.K), -1., 0)
        partial_xx = 0
        Lg = self.mu * x * partial_x + 0.5 * (self.sig ** 2) * (x ** 2) * partial_xx

        # Compute (Lg - rg)^-
        val = Lg - self.mu * g_val
        val_minus = -1 * where(val <= 0, val, 0)  # d2 x M

        # Compute f = -ru + (Lg - rg)^- * I
        return -self.mu * y + val_minus * indicator_ex

    def g(self, T, x, use_tensor=False):
        where = tf.where if use_tensor else np.where
        maximum = tf.math.maximum if use_tensor else np.maximum

        return maximum(self.K - x, 0) * where(x >= self.lower_barrier, 1, 0)
