import tensorflow as tf
from tensorflow.keras.layers import *


class CriticCNN(tf.keras.Model):

    def __init__(self, nTasks: int, nLast: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.nTasks = nTasks
        self.model = tf.keras.Sequential([
            Reshape(((nTasks+1)*nLast, 1), input_shape=((nTasks+1)*nLast, )),
            Conv1D(filters=3, kernel_size=(nTasks + 1), strides=(nTasks + 1), activation=tf.nn.relu),
            Flatten(),
            Dense(units=50, activation='relu'),
            Dense(units=30, activation='relu'),
            Dense(units=1, activation='relu')
        ])

    @tf.function
    def call(self, state):
        # tf.reshape(state, (1, (self.nTasks+1), ))
        return self.model(state)
