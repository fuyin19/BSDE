import numpy as np

from abc import ABC, abstractmethod


class FBSDE(ABC):
    """
    Simulate the forward SDE (1.1.2) with discretization scheme (3.1.2)
    Simulate the backward BSDE (1.1.1) with discretization scheme (3.1.2)

    d          dimension of the Brownian motion
    d1         dimension of X_t
    """

    def __init__(self, config, exclude_spot=False):
        self.config = config

        self.d = config.d
        self.d1 = config.d1
        self.d2 = config.d2
        self.exclude_spot = exclude_spot  # Decide if initial value is included
        self.dW = 0

    def draw(self, dW, x0, dt):
        """
        Inputs:
            dW: Gaussian increments with dimension [n_steps, d_bm, n_path]
            x0: The initial value X_0
            dt: size of time step for each increment in the time grid

        Output:
            The simulated result of X_t with dimension [n_steps, d_X, n_path].
            If initial stock price included, the dimension is [n_steps + 1, d_X, n_path].
        """
        self.dW = dW

        # Parameters
        N, d, M = dW.shape  # Get M, N, d

        t = 0  # Initial time
        x0 = np.array(x0)  # Initial value x0, d1
        x = np.tile(x0, (M, 1)).T  # Initial value x0 for all path, d1 x M

        # Prepare the Path matrix
        xs = np.zeros(shape=(N + 1, d, M))  # X_t path matrix, N+1 x d x M
        xs[0, :, :] = x  # Add Initial value x0 for all path

        # Compute X_t for all Path
        for (i, dw) in enumerate(dW):  # z, BM incre., d x M
            dw = dw.T.reshape(M, d, 1)  # z, BM incre., d x M -> M x d -> M x d x 1
            t += dt  # t, current time
            sig = self.sig_t(t, x).transpose((2, 0, 1))  # sig, vol matrix, d1 x d x M -> M x d1 x d

            x = x + self.mu_t(t, x) * dt + (sig @ dw)[:, :, 0].T  # current X_t, d1 x M

            xs[1 + i, :, :] = x  # Add current X_t for all path

        if self.exclude_spot:
            return xs[1:, :, :]
        else:
            return xs

    @abstractmethod
    def mu_t(self, t, x):
        """
        Drift in the forward SDE

        :param t: the current time
        :param x: the current value of X_t, d1 x M
        :return: mu_t, the drift value, d1 x M
        """
        pass

    @abstractmethod
    def sig_t(self, t, x):
        """
        Volatility in the forward SDE

        :param t: the current time
        :param x: the current value of X_t, d1 x M

        :return: sig_t, the volatility value, which is d1 x d x M
        """
        pass

    @abstractmethod
    def f(self, t, x, y, z, use_tensor=False):
        """
        Generator of the BSDE

        :param use_tensor: Determine tf.Tensor will be supported
        :param t: the current time
        :param x: the current X_t, d1 x M
        :param y: the current Y_t, d2 x M
        :param z: the current Z_t, d2 x d x M

        :return: f, the value of generator, d2 x M
        """
        pass

    @abstractmethod
    def g(self, T, x_T, use_tensor=False):
        """
        Final condition of the BSDE

        :param use_tensor: Determine tf.Tensor will be supported
        :param T: the final time
        :param x_T: the final value of X_T, d1 x M
        :return : g, the final value of Y_T, d2 x M
        """
        pass
