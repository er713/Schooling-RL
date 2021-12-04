import tensorflow as tf


class Critic(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=30, activation='relu'),
            tf.keras.layers.Dense(units=50, activation='relu'),
            tf.keras.layers.Dense(units=50, activation='relu'),
            tf.keras.layers.Dense(units=1, activation='relu')
        ])

    @tf.function
    def call(self, state):
        return self.model(state)