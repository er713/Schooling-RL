import tensorflow as tf
import numpy as np


class Actor(tf.keras.Model):

    def __init__(self, nTasks: int):
        super.__init__(self)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=100),
            tf.keras.layers.ReLu(),
            tf.keras.layers.Dense(units=200),
            tf.keras.layers.ReLu(),
            tf.keras.layers.Dense(units=200),
            tf.keras.layers.ReLu(),
            tf.keras.layers.Dense(units=nTasks)
        ])

    def call(self, state):
        return np.argmax(self.model(state))
