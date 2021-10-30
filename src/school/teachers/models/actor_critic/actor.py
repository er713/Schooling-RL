import tensorflow as tf
import numpy as np


class Actor(tf.keras.Model):

    def __init__(self, nTasks: int):
        super().__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=100, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=200, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=200, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=nTasks)
        ])

    def call(self, state):
        print("State: ",state)
        return self.model(state)
