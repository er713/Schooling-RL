import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, LayerNormalization
import numpy as np


class Actor(tf.keras.Model):

    def __init__(self, nTasks: int, verbose: bool = False, normalize: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.verbose = verbose
        if not normalize:
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(units=30, activation=tf.nn.relu),
                tf.keras.layers.Dense(units=50, activation=tf.nn.relu),
                tf.keras.layers.Dense(units=50, activation=tf.nn.relu),
                tf.keras.layers.Dense(units=nTasks)
            ])
        else:
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(units=30, activation=tf.nn.relu),
                LayerNormalization(),
                tf.keras.layers.Dense(units=50, activation=tf.nn.relu),
                LayerNormalization(),
                tf.keras.layers.Dense(units=50, activation=tf.nn.relu),
                LayerNormalization(),
                tf.keras.layers.Dense(units=nTasks)
            ])

    @tf.function
    def call(self, state):
        # if self.verbose:
        #     print("State: ", state)
        return self.model(state)
        
    def copy_weights(self, otherModel):
        model=otherModel.model
        for idx, layer in enumerate(model.layers):
            weights = layer.get_weights()
            self.model.layers[idx].set_weights(weights)
