import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, LayerNormalization
import numpy as np
from ...losses.actor_loss import actor_batch_loss


class Actor(tf.keras.Model):

    def __init__(self, nTasks: int, verbose: bool = False, normalize: bool = False, batch_norm: bool = False, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.verbose = verbose
        if batch_norm:
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(units=30, activation=tf.nn.relu),
                tf.keras.layers.Dense(units=50, activation=tf.nn.relu),
                BatchNormalization(),
                tf.keras.layers.Dense(units=50, activation=tf.nn.relu),
                BatchNormalization(),
                tf.keras.layers.Dense(units=nTasks)
            ])
        elif normalize:
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(units=30, activation=tf.nn.relu),
                LayerNormalization(),
                tf.keras.layers.Dense(units=50, activation=tf.nn.relu),
                LayerNormalization(),
                tf.keras.layers.Dense(units=50, activation=tf.nn.relu),
                LayerNormalization(),
                tf.keras.layers.Dense(units=nTasks)
            ])
        else:
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(units=30, activation=tf.nn.relu),
                tf.keras.layers.Dense(units=50, activation=tf.nn.relu),
                tf.keras.layers.Dense(units=50, activation=tf.nn.relu),
                tf.keras.layers.Dense(units=nTasks)
            ])

    @tf.function
    def call(self, state):
        # if self.verbose:
        #     print("State: ", state)
        return self.model(state)

    def copy_weights(self, otherModel):
        model = otherModel.model
        for idx, layer in enumerate(model.layers):
            weights = layer.get_weights()
            self.model.layers[idx].set_weights(weights)

    # @tf.function
    def train_step(self, states, actions, realQs):
        print("Retracing train_stepa@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("!!!!!!!!!!state", states, states.shape)
        print("!!!!!!!!!!!real!", realQs, realQs.shape)
        with tf.GradientTape() as tape:
            qPred = self(states, training=True)
            # qPred = tf.gather_nd(qPred, actions)
            lossValue = actor_batch_loss(qPred, actions, realQs)
        variables = self.model.trainable_variables
        grads = tape.gradient(lossValue, variables)
        self.optimizer.apply_gradients(zip(grads, variables))
