from typing import Iterable
import tensorflow as tf
from keras import Model, Sequential
from keras.layers import *


class ACNetwork(Model):
    def __init__(self, output_size: int, units: Iterable = (60, 50, 30), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq = Sequential([Dense(units=un, activation=tf.nn.relu) for un in units])
        self.seq.add(Dense(units=output_size))
        if output_size == 1:
            self.seq.add(ReLU())

    @tf.function
    def call(self, inputs, training=None, mask=None):
        return self.seq(inputs)

    def get_config(self):
        pass
