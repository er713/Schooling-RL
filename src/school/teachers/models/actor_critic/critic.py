import tensorflow as tf


class Critic(tf.keras.Model):

    def __init__(self):
        super.__init__(self)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=100, activation='relu'),
            tf.keras.layers.Dense(units=100, activation='relu'),
            tf.keras.layers.Dense(units=100, activation='relu'),
            tf.keras.layers.Dense(units=1, activation='relu')
        ])

    def call(self, state):
        return self.model(state)
