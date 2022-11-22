import numpy as np
import logging
import tensorflow as tf
import time
from bsde.LSMC.LSMC import generate_z_matrix

DELTA_CLIP = 50.0


class DeepBSDESolver(object):
    """
    The Deep BSDE method in the paper [EHJ17] and [HJE18]
    """
    def __init__(self, FBSDE, config_sim, config_NN, config_deepBSDE):
        self.FBSDE = FBSDE
        self.config_sim = config_sim
        self.config_deepBSDE = config_deepBSDE
        self.model = GlobalDNN(FBSDE=FBSDE, config_sim=config_sim, config_NN=config_NN)
        self.y_0 = self.model.y_0

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.PiecewiseConstantDecay(self.config_NN.lr_boundaries,
                                                                               self.config_NN.lr_values),
            epsilon=1e-8)

    def train_n_report(self):
        start_time = time.time()
        valid_data = self.FBSDE.draw(dW=np.sqrt(self.config_sim.dt) * generate_z_matrix(n_steps=self.config_sim.N,
                                                                                        n_paths=self.config_deepBSDE.valid_size,
                                                                                        d_bm=self.FBSDE.d),
                                     x0=self.config_sim.x0,
                                     dt=self.config_sim.dt), self.FBSDE.dW

        # begin sgd iteration
        for step in range(self.config_deepBSDE.num_iterations + 1):
            if step % self.config_deepBSDE.logging_frequency == 0:
                loss_val = self.loss(valid_data, training=False).numpy()
                y_0 = self.y_0.numpy()[0]
                elapsed_time = time.time() - start_time
                if self.config_deepBSDE.verbose:
                    logging.info("step: %5u,    loss: %.4e, Y0: %.4e,   elapsed time: %3u" % (
                        step, loss_val, y_0, elapsed_time))
            self.train_step(self.FBSDE.sample(self.config_deepBSDE.batch_size))

    def loss(self, FSDE_path, training):
        dW, x = FSDE_path
        y_terminal = self.model(dW, x, training)
        delta = y_terminal - self.FBSDE.g(self.FBSDE.T, x[:, :, -1])
        # use linear approximation outside the clipped range
        loss = tf.reduce_mean(tf.where(tf.abs(delta) < DELTA_CLIP, tf.square(delta),
                                       2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2))

        return loss

    def gradient(self, FSDE_path, training):
        with tf.GradientTape(persistent=True) as tape:
            loss_val = self.loss(FSDE_path, training)
        grad = tape.gradient(loss_val, self.model.trainable_variables)
        del tape
        return grad

    @tf.function
    def train(self, train_data):
        grad = self.gradient(train_data, training=True)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))


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
                               )  # 1 x d
        self.subnet = [FeedForwardNN(FBSDE.config, config_NN) for i in range(self.config_sim.N-1)]

    def call(self, dW, x, training):
        """
        Forward pass of the global Neural Net

        :param dW: BM incre., M x N x d
        :param x: X_path, M x N x d
        :param training: ?
        :return: y_T, the final value of y, d2
        """
        # y, M x 1
        # z, M x d
        # x_t, M x d
        # dW_t, M x d

        ts = np.arange(0, self.config_sim.N) * self.config_sim.dt
        all_one_vec = tf.ones(shape=[self.config_sim.M, self.FBSDE.d2], dtype=self.config_subnet.dtype)  # M x 1

        y = all_one_vec * self.y_0
        z = tf.matmul(all_one_vec, self.z_init)

        for t in range(0, self.config_sim.N - 1):
            x_t = x[:, t, :]
            dW_t = dW[:, t, :]

            y = y - self.config_sim.dt * self.FBSDE.f(ts[t], x_t, y, z) \
                + tf.reduce_sum(z * dW_t, 1, keepdims=True)

            z = self.subnet[t](x[:, t+1, :], training) / self.FBSDE.d

        # terminal time
        y = y - self.config_sim.dt * self.FBSDE.f(ts[-1], x[:, -2, :], y, z) \
            + tf.reduce_sum(z * dW[:, -1, :], 1, keepdims=True)

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




