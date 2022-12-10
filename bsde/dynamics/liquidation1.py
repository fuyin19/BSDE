import numpy as np
import tensorflow as tf
from bsde.dynamics.fbsde import FBSDE
from bsde.solver.FBSNNs import FBSNN


# Implement the dynamics
class HJB_liquidation1_FBSDE(FBSDE):
    """
    X_t for Liquidation without penalties
    """
    def __init__(self, config, exclude_spot=False):
        FBSDE.__init__(self, config=config, exclude_spot=exclude_spot)
        self.sig_s = config.sig_s
        self.eps = config.eps
        self.lb = config.lb
        self.k = config.k
        self.T = config.T

    def mu_t(self, t, x):
        return np.zeros(shape=x.shape)

    def sig_t(self, t, x):
        val1 = np.repeat(np.array([self.sig_s, 0])[:, np.newaxis],
                         x.shape[1],
                         axis=1)  # 2 x M
        val2 = np.array([[0, 0], [0, self.eps]]) @ x  # 2 x M

        return np.stack((val1, val2), axis=0)  # 2 x 2 x M

    def f(self, t, x, y, z, use_tensor=False):
        """
        Generator of Y_t

        :param use_tensor: tensorflow or numpy data structure
        :param t: Current time
        :param x: Current X_t, d1 x M, 2 x M
        :param y: Current Y_t, d2 x M, 1 x M
        :param z: Current Z_t, d2 x d x M, 1 x 2 x M
        :return: the value of generator, 1 x M
        """
        q = x[1:2, :]  # 1 x M
        s = x[0:1, :]  # 1 x M

        A = -1/(self.eps*q)
        neg_partial_q = A * z[:, 1, :]

        return 1/(4*self.k) * (s + neg_partial_q)**2

    def g(self, T, x_T, use_tensor=False):
        """
        final condition of Y_t

        :param use_tensor: tensorflow or numpy data structure
        :param T: final time
        :param x_T: final time value of X_T, 2 x M
        :return: final time value of Y_T, 1 x M
        """
        s_T = x_T[0:1, :]  # 1 x M
        q_T = x_T[1:2, :]  # 1 x M

        thresh = 0.9

        # return (q_T-thresh)*s_T - self.lb * ((q_T-thresh)**2)
        return q_T*s_T - self.lb * (q_T**2)


class HJB_liquidation1_solver(FBSNN):
    def __init__(self, config_dynamic, config_solver):
        self.sig_s = config_dynamic.sig_s
        self.eps = config_dynamic.eps
        self.lb = config_dynamic.lb
        self.k = config_dynamic.k
        super(HJB_liquidation1_solver, self).__init__(T=config_dynamic.T,
                                                      M=config_solver.M,
                                                      N=config_solver.N,
                                                      D=config_dynamic.d,
                                                      Xi=config_solver.x0,
                                                      layers=config_solver.layers)

    def phi_tf(self, t, X, Y, Z):
        """
        Generator of the BSDE

        :param t: M x 1
        :param X: M x d
        :param Y: M x 1
        :param Z: M x d
        :return: Generator, M x 1
        """
        q = X[:, 1:2]  # M x 1
        s = X[:, 0:1]  # M x 1

        A = -1 / (self.eps * q)  # M x 1
        neg_partial_q = A * Z[:, 1:2]  # M x 1

        return -1 / (4 * self.k) * (s + neg_partial_q) ** 2

    def g_tf(self, X):
        """
        Final condition

        :param X: M x d
        :return: Final condition, M x 1
        """
        q = X[:, 1:2]  # M x 1
        s = X[:, 0:1]  # M x 1

        return q * s - self.lb * (q ** 2)  # M x 1

    def mu_tf(self, t, X, Y, Z):
        """
        Drift of the Forward SDE

        :param t: M x 1
        :param X: M x d
        :param Y: M x 1
        :param Z: M x d
        :return: Drift, M x d
        """
        return tf.zeros(shape=X.shape, dtype='float32')

    def sigma_tf(self, t, X, Y):
        """
        Volatility of the Forward SDE

        :param t: M x 1
        :param X: M x d
        :param Y: M x 1
        :return: Vol, M x d x d
        """
        val1 = tf.repeat(tf.constant([self.sig_s, 0], dtype='float32')[None, :],
                         X.shape[0],
                         axis=0)  # M x 2
        val2 = tf.matmul(X, tf.constant([[0, 0], [0, self.eps]], dtype='float32'))  # M x 2

        return tf.stack((val1, val2), axis=1)  # M x 2 x 2
