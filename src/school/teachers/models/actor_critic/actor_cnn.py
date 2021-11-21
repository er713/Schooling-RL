import tensorflow as tf
from tensorflow.keras.layers import *


class ActorCNN(tf.keras.Model):

    def __init__(self, nTasks: int, nLast: int, verbose: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert nLast is not None
        self.verbose = verbose
        self.model = tf.keras.Sequential([
            Reshape(((nTasks + 1) * nLast, 1), input_shape=((nTasks + 1) * nLast,)),
            Conv1D(filters=3, kernel_size=(nTasks + 1), strides=(nTasks + 1), activation=tf.nn.relu),
            Flatten(),
            Dense(units=50, activation=tf.nn.relu),
            Dense(units=50, activation=tf.nn.relu),
            Dense(units=nTasks)
        ])

    @tf.function
    def call(self, state):
        if self.verbose:
            print("State: ", state)
        return self.model(state)
