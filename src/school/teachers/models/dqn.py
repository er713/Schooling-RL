import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from ..losses.dqn_loss import dqn_loss
from ..utils.dqn_structs import BatchRecord


class DQN(tf.keras.Model):

    def __init__(self, inputSize):
        super().__init__(name='dqn')
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.loss = dqn_loss()

        self.model = tf.keras.Sequential([
            Dense(3 * inputSize, activation='relu', input_shape=(None, inputSize)),
            Dense(2 * inputSize, activation='relu'),
            Dense(inputSize)
        ])

    #@tf.function
    def call(self, inputs, training):
        """ Given a state, return Q values of actions"""
        return self.model(inputs, training=training)

    def copy_weights(self, otherModel):
        for idx, layer in enumerate(otherModel.layers):
            weights = otherModel.layers.get_weights()
            self.model.layers[idx].set_weights(weights)

    #@tf.function
    def train_step(self, states, actions, realQs):
        with tf.GradientTape() as tape:
            qPred=self(states, training=True)
            qPred=tf.gather_nd(qPred, actions)
            lossValue = self.loss(realQs, qPred)
        variables = self.model.trainable_variables
        grads=tape.gradient(lossValue, variables)
        self.optimizer.apply_gradients(zip(grads, variables))
