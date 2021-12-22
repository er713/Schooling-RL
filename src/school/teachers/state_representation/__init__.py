import tensorflow as tf
__state = tf.Variable([], dtype=tf.float32, trainable=False, shape=(None,))
__state_all = tf.Variable([[], []], dtype=tf.float32, trainable=False, shape=(2, None))
from .n_last import get_state_normal, get_state_inverse
from .table import TableTeacher
from .all_history import get_state_inverse as get_state_history, get_state_for_rnn

__all__ = ["get_state_normal", "get_state_inverse", "TableTeacher", "get_state_history", "get_state_for_rnn"]
