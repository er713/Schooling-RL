import tensorflow as tf


class EmbeddedTasks(tf.keras.Model):
    def __init__(self, nTasks: int, embedding_size: int, required_history: int):
        super(EmbeddedTasks, self).__init__()
        self.nTasks = nTasks
        self.embedding_size = embedding_size
        self.required_history = required_history
        self.null_task_id = tf.constant([self.nTasks])
        # Trainable values
        self.task_emb = tf.keras.layers.Embedding(nTasks + 1, embedding_size, name='EmbeddedTasksIDLayer')
        self.null_mark_emb = tf.keras.layers.Embedding(1, 1, name='MarkForNoTask')

    def call(self, st) -> tf.Tensor:
        # st = get_state_inverse(results, idStudent, self.nTasks, shift)
        for _ in range(self.required_history - st.shape[1]):
            st = tf.concat([st, [self.null_task_id, self.null_mark_emb(0)]], 1)
        emb_tasks = self.task_emb(st[0])
        # tf.print('emb_var', self.task_emb.variables)
        res = tf.expand_dims(tf.concat([emb_tasks, tf.reshape(st[1], (-1, 1))], 1), 0)
        # tf.print('emb_res', res)
        return res
