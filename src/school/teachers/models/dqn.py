import tensorflow as tf


class DQN(tf.keras.Model):

    def __init__(self):
        super().__init__(name='dqn')
        pass

    def train_step(self):
        """Update weight"""
        pass

    def call(self, inputs, training):
        """ Given a state, return Q values of actions"""
        return self.model(inputs, training=training)
