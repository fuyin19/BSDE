import numpy as np

from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR


def generate_z_matrix(n_paths, n_steps, d_bm, seed=42):
    """
    Function for generating a matrix of iid standard normal random variables

    Inputs:
        n_paths: number of paths
        n_steps: number of time steps/dt in each path
        d_bm: dimension of the Brownian motion increments

    Output:
        a normalized matrix of normal random variables with dimension (M x N x d), where
        M = number of path
        N = number total time steps in each path
        d_bm = dimension of the Brownian motion increments
    """

    np.random.seed(seed)

    Zs = np.random.normal(size=n_paths * n_steps * d_bm)  # N x d x M, iid standard normal
    Zs = (Zs - Zs.mean()) / Zs.std()  # Normalize to standard normal
    Z_matrix = Zs.reshape(n_steps, d_bm, n_paths)  # N x d x M after reshape

    return Z_matrix


class LSMC(ABC):
    """
    Least Square Monte Carlo Method for FBSDE

    Y_t, the BSDE
    X_t, the FSDE
    """

    def __init__(self, Y_t, X_t, dZ, x0, dt, basis_funcs=None, **kwargs):
        """
        Initialize the LSMC solver

        :param Y_t: the backward dynamics
        :param X_t: the forward dynamics
        :param dZ: the unscaled BM increments
        :param x0: the initial value of X_t
        :param dt: time increments
        :param basis_funcs: list of feature map for X_t, constant function should always be included
        as the first element of the list.
        :param kwargs: additional arguments, i.e. predefined basis function type
        """
        self.X_t = X_t                 # FSDE dynamics
        self.Y_t = Y_t                 # BSDE dynamics
        self.dZ = dZ                   # BM increments,
        self.N, self.d, self.M = dZ.shape            # M, N, d
        self.d1, self.d2 = x0.shape[0], self.Y_t.d2  # d1, d2

        self.x0 = x0                # Initial value of X_t
        self.dt = dt                # Time increments

        if not basis_funcs:
            basis_funcs_type = kwargs.get('basis_funcs_type', 'poly')
            if basis_funcs_type == 'poly':
                highest_degree = kwargs.get('highest_degree', 3)
                self.basis = [lambda x, coef=i: x**coef for i in range(highest_degree + 1)]
                self.kn = highest_degree + 1
                self.n_features = (self.kn-1)*self.d1 + 1

            elif basis_funcs_type == 'trig':
                n_freq = kwargs.get('n_freq', 5)
                self.basis = [lambda x: x**0]
                for i in range(1, n_freq+1):
                    self.basis.append(lambda x, coef=i: np.cos(coef*x))
                    self.basis.append(lambda x, coef=i: np.sin(coef*x))

                self.kn = 1 + 2*n_freq
                self.n_features = (self.kn - 1) * self.d1 + 1

                # [lambda x:x**0, lambda x:np.cos(x), lambda x:np.cos(2*x), lambda x:np.cos(3*x),
                # lambda x:np.cos(4*x), lambda x:np.cos(5*x), lambda x:np.cos(6*x),
                # lambda x: np.sin(x), lambda x: np.sin(2*x), lambda x: np.sin(3*x),
                # lambda x:np.sin(4*x), lambda x:np.sin(5*x), lambda x:np.sin(6*x)]
        else:
            self.kn = len(basis_funcs)  # Number of basis function
            self.n_features = (self.kn - 1) * self.d1 + 1  # Number of features in the feature map
            self.basis = basis_funcs    # List of basis functions

        self.X_path = self.X_t.draw(self.dZ, self.x0, self.dt)                # X_t path, N+1 x d x M
        self.Y_path = np.zeros(shape=(self.N, self.d2, self.M))               # Y_t path, N x d2 x M
        self.Z_path = np.zeros(shape=(self.N - 1, self.d2, self.d, self.M))           # Z_t path, N-1 x d2 x d x M

        self.T = self.N * self.dt  # Maturity
        self.y0 = 0  # Initial value of Y_t
        self.z0 = 0  # Initial value of Z_t

    def basis_transform(self, x):
        """
        Compute the data matrix X after feature map transformation of current X_t for all path

        :param x: the current value of X_t for all path, d1 x M
        :return: X, the data matrix, M x ((k_n-1) x d1 + 1). X = [1, f_2(x_t).T, f_3(x_t).T,..., f_kn(x_t).T]
        """
        X = np.zeros(shape=(self.M, self.n_features))

        for (i, f) in enumerate(self.basis):
            if i == 0:
                X[:, 0] = np.ones(self.M)        # M
            else:
                X[:, self.d1*(i-1)+1:self.d1*i+1] = f(x).T  # M x d1
        return X

    @ abstractmethod
    def fit(self, n, x, y):
        """
        Compute Z_t and Y_t approximation at current time using basis functions of X_t alongside with
            1. alpha, Model parameter for Z_t approximation
            2. beta, Model parameter for Y_t approximation

        Input:
            n, Current time step
            x, X_n, d1 x M
            y, Y_n+1, d2 x M
        Output:
            z, Z_t at current time, d2 x d x M
            y, Y_t at current time, d2 x M
        """
        pass

    def est_z(self, y, dz):
        """
        Compute target variable z in the regression

        Input:
            y, current y, d2 x M
            dz, BM incre. before scaling, d x M
        Output:
            z_estimation, d2 x d x M
        """
        if self.d1 == 1 and self.d == 1:
            val = 1 / np.sqrt(self.dt) * (np.array([y * dz]))
        else:
            val = 1 / np.sqrt(self.dt) * np.matmul(y.T.reshape(self.M, self.d2, 1),
                                                   dz.T.reshape(self.M, 1, self.d)).transpose((1, 2, 0))

        return val

    def est_y(self, t, x, y, z):
        """
        Compute target variable y in the regression

        Input:
            x, d1 x M
            y, d2 x M
            z, d2 x d x M
        Output:
            y_estimation, d2 x M
        """
        return y + self.Y_t.f(t, x, y, z) * self.dt

    def solve(self):
        """
        Backward Propagation for finding alpha, beta, y, z at each discretized time step t_n
        """

        # n=N
        x = self.X_path[-1, :, :]  # Final value X_T, d1 x M
        y = self.Y_t.g(self.T, x)  # Final value Y_T, d2 x M
        self.Y_path[-1, :, :] = y

        # n = N-1,...,1 -> t_N-1,...,t_1
        for n in range(self.N - 1, 0, -1):

            x = self.X_path[n, :, :]    # X_n, d1 x M

            # Compute Z_n, Y_n
            z, y = self.fit(n, x, y)    # d2 x d x M, d2 x M

            # Record value
            self.Z_path[n - 1, :, :, :] = z
            self.Y_path[n - 1, :, :] = y

        self.z0 = np.mean(self.est_z(y, self.dZ[0, :, :]), axis=2)  # d2 x d, d2 x d x M along M
        self.y0 = np.mean(self.est_y(0, self.x0, y, self.z0), axis=1)  # d2, d2 x M along M


class LSMC_linear(LSMC):
    """
    Least Square Monte-Carlo method for solving FBSDE, where Y_t and Z_t are approximated by linear combination
    of basis functions of X_t
    """
    def __init__(self, Y_t, X_t, dZ, x0, dt, reg_method=None, basis_funcs=None, **kwargs):
        super().__init__(Y_t, X_t, dZ, x0, dt, basis_funcs, **kwargs)

        self.alphas = np.zeros(shape=(self.N - 1, self.n_features, self.d2, self.d))  # All alphas, N-1 x kn x d2 x d
        self.betas = np.zeros(shape=(self.N - 1, self.n_features, self.d2))           # All betas, N-1 x kn x d2

        self.reg_method = reg_method

        if reg_method == 'lasso':
            lb = kwargs.get('lb', 1)
            self.model = Lasso(alpha=lb, fit_intercept=False)
        elif reg_method == 'ridge':
            lb = kwargs.get('lb', 1)
            self.model = Ridge(alpha=lb, fit_intercept=False)
        elif reg_method == 'elastic_net':
            lb = kwargs.get('lb', 1)
            l1_ratio = kwargs.get('l1_ratio', 0.5)
            self.model = ElasticNet(alpha=lb, l1_ratio=l1_ratio, fit_intercept=False)
        else:
            self.model = LinearRegression(fit_intercept=False)

    def fit(self, n, x, y):
        """
        Compute (alpha, Z_t), (beta, Y_t) at current time using linear model

        Input:
            n, Current time step
            x, X_n, d1 x M
            y, Y_n+1, d2 x M
        Output:
            alpha, the linear coefficient of Z_t at current time step n, k_n x d2 x d
            z, Z_t at current time, d2 x d x M
            beta, the linear coefficient of Y_t at current time step n, k_n x d2
            y, Y_t at current time, d2 x M
        """
        t_n = n * self.dt               # Current time at n
        dZ_n = self.dZ[n, :, :]         # Brownian motion increments before scaling, d x M
        X = self.basis_transform(x)     # M x kn

        # Compute (alpha, z)
        z_fit = self.est_z(y, dZ_n).transpose((2, 0, 1)).reshape(self.M, self.d2 * self.d)  # d2 x d x M -> M x d2 x d ->  M x (d2 x d)
        model_z = self.model.fit(X, z_fit)

        alpha = model_z.coef_.T.reshape(self.n_features, self.d2, self.d)  # (d2 x d) x n_f -> n_f x (d2 x d) -> n_f x d2 x d
        z = model_z.predict(X).reshape(self.M, self.d2, self.d).transpose((1, 2, 0))  # M x (d2 x d) -> M x d2 x d -> d2 x d x M

        # Compute (beta, y)
        y_fit = self.est_y(t_n, x, y, z).T  # d2 x M -> M x d2
        model_y = self.model.fit(X, y_fit)

        beta = model_y.coef_.T     # d2 x k_n -> k_n x d2
        y = model_y.predict(X).T   # M x d2 -> d2 x M

        # record value
        self.alphas[n - 1, :, :, :] = alpha.reshape(self.n_features, self.d2, self.d)
        self.betas[n - 1, :, :] = beta.reshape(self.n_features, self.d2)

        return z, y


class LSMC_svm(LSMC):
    """
    Least Square Monte-Carlo method for solving FBSDE, where Y_t and Z_t are approximated by linear combination
    of basis functions of X_t
    """
    def __init__(self, Y_t, X_t, dZ, x0, dt, basis_funcs=None, **kwargs):
        super().__init__(Y_t, X_t, dZ, x0, dt, basis_funcs, **kwargs)

        self.kernel = kwargs.get('kernel', 'linear')
        self.tol = kwargs.get('tol', 1e-3)

    def fit(self, n, x, y):
        """
        Compute (alpha, Z_t), (beta, Y_t) at current time using linear model

        Input:
            n, Current time step
            x, X_n, d1 x M
            y, Y_n+1, d2 x M
        Output:
            alpha, alpha at current time, k_n x d2 x d
            z, Z_t at current time, d2 x d x M
            beta, beta at current time, k_n x d2
            y, Y_t at current time, d2 x M
        """
        t_n = n * self.dt            # Current time at n
        dZ_n = self.dZ[n, :, :]      # Brownian motion increments before scaling, d x M
        X = self.basis_transform(x)  # M x kn

        # Compute (alpha, z_n)
        z_fit = self.est_z(y, dZ_n).transpose((2, 0, 1)).reshape((self.M, self.d2 * self.d))  # d2 x d x M -> M x d2 x d ->  M x (d2 x d)
        z = np.zeros(shape=(self.d2 * self.d, self.M))                                        # (d2 x d) x M

        for i in range(self.d2 * self.d):
            SVR_z = SVR(kernel=self.kernel, tol=self.tol).fit(X, z_fit[:, i])
            z_i = SVR_z.predict(X)                                              # M
            z[i, :] = z_i
        z = np.reshape(z, (self.d2, self.d, self.M))                            # d2 x d x M

        # Compute (beta, y_n)
        y_fit = self.est_y(t_n, x, y, z).T        # d2 x M -> M x d2
        y = np.zeros(shape=(self.d2, self.M))     # d2 x M

        for i in range(self.d2):
            SVR_y = SVR(kernel=self.kernel, tol=self.tol).fit(X, y_fit[:, i])
            y_i = SVR_y.predict(X)                # M
            y[i, :] = y_i

        return z, y
