import tensorflow as tf


class Critic(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=30, activation='relu', name='critic1'),
            tf.keras.layers.Dense(units=50, activation='relu', name='critic2'),
            tf.keras.layers.Dense(units=50, activation='relu', name='critic3'),
            tf.keras.layers.Dense(units=1, activation='relu', name='critic_out')
        ])

    @tf.function
    def call(self, state):
        return self.model(state)
