import tensorflow as tf
from tensorflow.keras.layers import *
from ...layers import NLastHistoryCNN


class CriticCNN(tf.keras.Model):

    def __init__(self, nTasks: int, nLast: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.nTasks = nTasks
        self.model = tf.keras.Sequential([
            NLastHistoryCNN(nTasks, nLast),
            Dense(units=50, activation='relu'),
            Dense(units=30, activation='relu'),
            Dense(units=1, activation='relu')
        ])

    @tf.function
    def call(self, state):
        # tf.reshape(state, (1, (self.nTasks+1), ))
        return self.model(state)
