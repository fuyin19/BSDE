import numpy as np
from FBSDE import *


# Implement the FBSDE
class X_t(FSDE):
    """
    X_t for Liquidation without penalties
    """
    def __init__(self, sig_s, eps, exclude_spot=False):
        super().__init__(d=2, d1=2, exclude_spot=exclude_spot)
        self.sig_s = sig_s
        self.eps = eps

    def mu_t(self, t, x):
        return np.zeros(shape=x.shape)

    def sig_t(self, t, x):
        sig = np.array([[self.sig_s, 0], [0, self.eps]]).reshape((2, 2, 1))  # 2 x 2
        return np.repeat(sig, x.shape[1], axis=2)                            # 2 x 2 x M
           #[:, :, np.newaxis]


class Y_t(BSDE):
    """
    Y_t for Liquidation without penalties
    """
    def __init__(self, eps, k):
        super().__init__(d2=1)
        self.eps = eps
        self.k = k

    def f(self, t, x, y, z):
        """
        Generator of Y_t

        :param t: Current time
        :param x: Current X_t, d1 x M, 2 x M
        :param y: Current Y_t, d2 x M, 1 x M
        :param z: Current Z_t, d2 x d x M, 1 x 2 x M
        :return: the value generator, 1 x 1 x M
        """
        s = x[0:1, :]                                        # 1 x M
        b = np.array([[1/2, 1/2]]).T                         # 2 x 1
        A = np.array([[0, 1/self.eps], [0, 1/self.eps]])     # 2 x 2

        const = 1/(4*self.k)
        val = (s - np.einsum('jik,li->ljk', z, b.T @ A)[0])**2  # 1 x 2 x M, 1 x 2 -> 1 x 1 x M -> 1 x M

        return const * val                                      # 1 x M

    def g(self, T, x_T):
        """
        final condition of Y_t

        :param T: final time
        :param x_T: final time value of X_T, 2 x M
        :return: final time value of Y_T, 1 x M
        """
        final = np.zeros(shape=(self.d2, x_T.shape[1]))
        mask = x_T[1:, :] > 0
        final[mask] = -10000000

        return final
