import tensorflow as tf
from tensorflow.keras.layers import *


class NLastHistoryCNN(tf.keras.Model):
    def __init__(self, nTasks: int, nLast: int, filters: int = 3):
        super(NLastHistoryCNN, self).__init__()
        self.layer = tf.keras.Sequential([
            Reshape(((nTasks + 1) * nLast, 1), input_shape=((nTasks + 1) * nLast,)),
            Conv1D(filters=filters, kernel_size=(nTasks + 1), strides=(nTasks + 1), activation=tf.nn.relu),
            Flatten()
        ])

    def call(self, state):
        return self.layer(state)
