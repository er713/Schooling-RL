from typing import List
import tensorflow as tf
from tensorflow.keras.layers import *
from ..layers import EmbeddedTasks


class RNNWrapper(tf.keras.Model):
    def __init__(self, rnn_units, nTasks, task_embedding_size, models: List):
        super().__init__(self)
        self.task_embeddings = EmbeddedTasks(nTasks, task_embedding_size, 1)

        self.h01 = Embedding(1, rnn_units,
                             name='emb_h01')  # , embeddings_constraint=tf.keras.constraints.MinMaxNorm(-1., 1.))
        self.h02 = Embedding(1, rnn_units, name='emb_h02')
        self.rnn = LSTMCell(rnn_units)

        self.models = models

    def _call_rnn(self, task_with_mark, student_state=None):
        if student_state is None:
            student_state = [tf.expand_dims(self.h01(0), 0),
                             tf.expand_dims(self.h02(0), 0)]
        embedded_tasks = self.task_embeddings(task_with_mark)
        embedded_tasks = tf.reshape(embedded_tasks, (1, -1))
        return self.rnn(embedded_tasks, student_state)

    def call(self, task_with_mark, student_state=None):
        out, state = self._call_rnn(task_with_mark, student_state)
        results = []
        for m in self.models:
            results.append(m(out))
        return results, state

    def get_specific_call(self, task_with_mark, student_state=None, modelId: int = 0):
        out, state = self._call_rnn(task_with_mark, student_state)
        result = self.models[modelId](out)
        return result, state

    def get_specific_variables(self, modelId: int = 0):
        # variables = self.trainable_variables
        emb_rnn = []
        [[emb_rnn.append(l) for l in layer.trainable_variables] for layer in
         (self.task_embeddings, self.h01, self.h02, self.rnn)]
        sec = self.models[modelId].trainable_variables
        return [*emb_rnn, *sec]
