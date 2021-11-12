import tensorflow as tf


def dqn_loss():
    return tf.keras.losses.Huber()
