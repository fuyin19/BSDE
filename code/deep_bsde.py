import numpy as np
import tensorflow as tf


class DeepBSDESolver(object):
    """
    The Deep BSDE method in the paper [EHJ17] and [HJE18]
    """
    def __init__(self, FBSDE, config_sim):
        self.FBSDE = FBSDE
        self.config_sim = config_sim
        self.model = GlobalDNN(FBSDE=FBSDE, config_sim=config_sim)


class GlobalDNN(tf.keras.Model):
    """
    Global deep neural net architecture for the deep BSDE method
    """
    def __init__(self, FBSDE, config_sim, config_NN):
        super(GlobalDNN, self).__init__()
        self.FBSDE = FBSDE
        self.config_NN = config_NN
        self.config_sim = config_sim
        self.y_0 = tf.Variable(np.random.uniform(low=self.config_NN.y_init_range[0],
                                                 high=self.config_NN.y_init_range[1],
                                                 size=[self.FBSDE.d2])
                               )
        self.z_0 = tf.Variable(np.random.uniform(low=-.1, high=.1,
                                                 size=[self.FBSDE.d2, self.FBSDE.d])
                               )
        self.subnet = [FeedForwardNN(FBSDE.config, config_NN) for i in range(self.config_sim.N-1)]

    def call(self, dW, x, training):
        """

        :param dW: M x N x d
        :param x:
        :param training:
        :return:
        """

        ts = np.arange(0, self.config_sim.N) * self.config_sim.dt
        all_one_vec_y = tf.ones(shape=[self.config_sim.M, self.FBSDE.d2], dtype=self.config_subnet.dtype)
        all_one_vec_z = tf.ones(shape=[self.config_sim.M, self.FBSDE.d2, self.FBSDE.d], dtype=self.config_subnet.dtype)

        y = all_one_vec_y * self.y_0  # M x d2
        z = all_one_vec_z * self.z_0  # M x d2 x d

        for t in range(0, self.config_sim.N - 1):
            x_t = x[:, t, :]          # M x d1
            dW_t = dW[:, t, :, None]  # M x d x 1

            y = y - self.config_sim.dt * self.FBSDE.f(ts[t], x_t, y, z) + tf.matmul(z, dW_t)[:, :, 0]
            z = self.subnet[t](x[:, t+1, :], training) / self.FBSDE.d

        # terminal time
        dW_T = dW[:, -1, :, None]
        y = y - self.config_sim.dt * self.FBSDE.f(ts[-1], x[:, -2, :], y, z) + tf.matmul(z, dW_T)[:, :, 0]

        return y


class FeedForwardNN(object):
    """
    The subnets (approximating Z_t) for the global neural network of deep BSDE method
    """
    def __init__(self, config_FBSDE, config_NN):
        super(FeedForwardNN, self).__init__()

        d = config_FBSDE.d

        self.n_hiddens = config_NN.n_hiddens
        self.n_total = len(self.n_hiddens) + 2

        self.batch_normalization_layers = [
            tf.keras.layers.BatchNormalization(
                momentum=0.99,
                epsilon=1e-6,
                beta_initializer=tf.random_normal_initializer(0.0, stddev=0.1),
                gamma_initializer=tf.random_uniform_initializer(0.1, 0.5)
            )
            for i in range(self.n_total)]

        self.dense_layers = [tf.keras.layers.Dense(self.n_hiddens[i],
                                                   use_bias=False,
                                                   activation=None)
                             for i in range(len(self.n_hiddens))]

        self.dense_layers.append(tf.keras.layers.Dense(d, activation=None))

    def call(self, x, training):
        """
        structure: bn -> (dense -> bn -> relu) * len(num_hiddens) -> dense -> bn
        """
        x = self.batch_normalization_layers[0](x, training)
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            x = self.batch_normalization_layers[i + 1](x, training)
            x = tf.nn.relu(x)
        x = self.dense_layers[-1](x)
        x = self.batch_normalization_layers[-1](x, training)
        return x




