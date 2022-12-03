import numpy as np
import tensorflow as tf
import time
from bsde.solver.lsmc import generate_z_matrix
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
print("Using TensorFlow version %s" % tf.__version__)

DELTA_CLIP = 50.0


class DeepBSDESolver(object):
    """
    The Deep BSDE method in the paper [EHJ17] and [HJE18]
    """
    def __init__(self, FBSDE, config_deep_bsde):
        self.FBSDE = FBSDE
        self.cfg = config_deep_bsde
        self.model = GlobalDNN(FBSDE=FBSDE, config_deep_bsde=config_deep_bsde)
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.PiecewiseConstantDecay(self.cfg.lr_boundaries,
                                                                               self.cfg.lr_values),
            epsilon=1e-8)

        self.y0 = self.model.y0
        self.dW = tf.cast(np.sqrt(self.cfg.dt) * generate_z_matrix(n_paths=self.cfg.M,
                                                                   n_steps=self.cfg.N,
                                                                   d_bm=self.FBSDE.d,
                                                                   seed=self.cfg.seed), dtype=self.cfg.dtype)  # N x d x M
        self.X_path = tf.cast(self.FBSDE.draw(dW=self.dW,
                                              x0=self.cfg.x0,
                                              dt=self.cfg.dt), dtype=self.cfg.dtype)  # N x d x M
        self.training_time = 0

    def train(self):
        start = time.time()
        for step in range(self.cfg.n_iterations + 1):
            # Report training result
            if step % self.cfg.report_freq == 0:
                print('step: {}, time: {}, y_0: {}'.format(step, int(time.time()-start), self.y0[0]))

            # Current batch
            dW = self.dW[:, :, step*self.cfg.batch_size:(step+1)*self.cfg.batch_size]
            X_path = self.X_path[:, :, step*self.cfg.batch_size:(step+1)*self.cfg.batch_size]

            self.train_step((dW, X_path))
        self.training_time = time.time() - start

    def loss(self, inputs, training):
        dW, X_path = inputs
        y_T = self.model((dW, X_path), training)
        delta = y_T - self.FBSDE.g(self.FBSDE.T, X_path[-1, :, :], use_tensor=True).T
        # use linear approximation outside the clipped range
        loss_val = tf.reduce_mean(tf.where(tf.abs(delta) < DELTA_CLIP, tf.square(delta),
                                  2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2))

        return loss_val

    def gradient(self, inputs, training):
        with tf.GradientTape(persistent=True) as tape:
            loss_val = self.loss(inputs, training)
        grad = tape.gradient(loss_val, self.model.trainable_variables)
        del tape
        return grad

    @tf.function
    def train_step(self, train_data):
        grad = self.gradient(train_data, training=True)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))


class GlobalDNN(tf.keras.Model):
    """
    Global deep neural net architecture for the deep BSDE method
    """
    def __init__(self, FBSDE, config_deep_bsde):
        super(GlobalDNN, self).__init__()
        self.FBSDE = FBSDE
        self.cfg = config_deep_bsde
        self.y0 = tf.Variable(np.random.uniform(low=self.cfg.y_init_range[0],
                                                high=self.cfg.y_init_range[1],
                                                size=[self.FBSDE.d2])
                              )
        self.z0 = tf.Variable(np.random.uniform(low=-.1, high=.1,
                                                size=[self.FBSDE.d2, self.FBSDE.d])
                              )  # 1 x d
        self.subnet = [FeedForwardNN(self.FBSDE, self.cfg) for _ in range(self.cfg.N-1)]

    def call(self, inputs, training):
        """
        Forward pass of the global Neural Net

        :param inputs: dW, X_path;
            dW: BM incre., N x d x M
            X_path: Forward path, N x d x M
        :param training: Trainable, True/False
        :return: y_T, the final value of y, d2
        """
        dW, X_path = inputs
        ts = np.arange(0, self.cfg.N) * self.cfg.dt
        all_one_vec = tf.ones(shape=[self.cfg.batch_size, self.FBSDE.d2], dtype=self.cfg.dtype)  # M x 1

        y = all_one_vec * self.y0  # M x 1
        z = tf.matmul(all_one_vec, self.z0)  # d x M

        for t in range(0, self.cfg.N - 1):
            y = y - self.cfg.dt * self.FBSDE.f(ts[t], X_path[t, :, :], y.T, z, use_tensor=True).T \
                + tf.reduce_sum(z * dW[t, :, :], 0, keepdims=True).T  # d x M -> 1 x M -> M x 1

            z = self.subnet[t](X_path[t+1, :, :], training) / self.FBSDE.d

        # terminal time
        y = y - self.cfg.dt * self.FBSDE.f(ts[-1], X_path[-2, :, :], y.T, z, use_tensor=True).T \
            + tf.reduce_sum(z * dW[-1, :, :], 0, keepdims=True).T

        return y


class FeedForwardNN(tf.keras.Model):
    """
    The subnets (approximating Z_t) for the global neural network of deep BSDE method
    """
    def __init__(self, FBSDE, config_deep_bsde):
        super(FeedForwardNN, self).__init__()

        d = FBSDE.d

        self.n_hiddens = config_deep_bsde.n_hiddens
        self.n_total = len(self.n_hiddens) + 2

        self.batch_normalization_layers = [
            tf.keras.layers.BatchNormalization(
                momentum=0.99,
                epsilon=1e-6,
                beta_initializer=tf.random_normal_initializer(0.0, stddev=0.1),
                gamma_initializer=tf.random_uniform_initializer(0.1, 0.5)
            )
            for _ in range(self.n_total)]

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
