import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import *


class AllHistoryCNN(tf.keras.Model):
    def __init__(self, embedded_len: int, out_dim: int, filters: int = 5):
        super(AllHistoryCNN, self).__init__()
        # self.out_dim = out_dim
        self.out_dim = out_dim
        self.filters = filters
        self.conv = Conv1D(filters=filters, kernel_size=1, strides=1, activation=tf.nn.relu,
                           input_shape=(None, embedded_len + 1))
        self.seq = tf.keras.Sequential([
            tfa.layers.AdaptiveMaxPooling1D(output_size=out_dim),
            Flatten()
        ])

    def call(self, state):
        # print('cnn_state:', state)
        af_conv = self.conv(state)
        # print(state.shape, af_conv.shape)
        rest = af_conv.shape[1] % self.out_dim
        af_conv = tf.concat([af_conv, tf.fill((1, (self.out_dim - rest), self.filters), -1e6)], 1)
        # print(af_conv.shape)
        res = self.seq(af_conv)
        # print('cnn_res:', res)
        # tf.print('cnn_res:', res)
        return res
