import tensorflow as tf
from tensorflow.keras.layers import *
from ...layers import NLastHistoryCNN


class ActorCNN(tf.keras.Model):

    def __init__(self, nTasks: int, nLast: int, verbose: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert nLast is not None
        self.verbose = verbose
        self.model = tf.keras.Sequential([
            NLastHistoryCNN(nTasks, nLast),
            Dense(units=50, activation=tf.nn.relu),
            Dense(units=50, activation=tf.nn.relu),
            Dense(units=nTasks)
        ])

    @tf.function
    def call(self, state):
        if self.verbose:
            print("State: ", state)
        return self.model(state)
