import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, LayerNormalization


class Critic(tf.keras.Model):

    def __init__(self, normalize: bool = False, batch_norm: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if batch_norm:
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(units=30, activation='relu', name='critic1'),
                tf.keras.layers.Dense(units=50, activation='relu', name='critic2'),
                BatchNormalization(),
                tf.keras.layers.Dense(units=50, activation='relu', name='critic3'),
                BatchNormalization(),
                tf.keras.layers.Dense(units=1, activation='relu', name='critic_out')
            ])
        elif normalize:
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(units=30, activation='relu', name='critic1'),
                LayerNormalization(),
                tf.keras.layers.Dense(units=50, activation='relu', name='critic2'),
                LayerNormalization(),
                tf.keras.layers.Dense(units=50, activation='relu', name='critic3'),
                LayerNormalization(),
                tf.keras.layers.Dense(units=1, activation='relu', name='critic_out')
            ])
        else:
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(units=30, activation='relu', name='critic1'),
                tf.keras.layers.Dense(units=50, activation='relu', name='critic2'),
                tf.keras.layers.Dense(units=50, activation='relu', name='critic3'),
                tf.keras.layers.Dense(units=1, activation='relu', name='critic_out')
            ])

    @tf.function
    def call(self, state):
        return self.model(state)
