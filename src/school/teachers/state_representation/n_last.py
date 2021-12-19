from typing import List, Dict
from . import tf, __state


def _complete_empty(_state: tf.Variable, nLast: tf.Tensor, nTasks: tf.Tensor) -> None:
    _state.assign(tf.concat(
        [_state, tf.zeros(shape=(tf.cast((nTasks + 1) * nLast - len(_state.value()), tf.int32)), dtype=tf.float32)], 0))


def get_state_inverse(results: Dict, idStudent: int, nLast: int, nTasks: int, shift: int = 0) -> tf.Tensor:
    """
    Function for getting state out of history (results) for specified student (in order t-1, t-2, ...).
    :param results: Dictionary with list of Results assigned to (int) Student ID
    :param idStudent: Student ID
    :param nTasks: Number of Tasks (during learning process)
    :param nLast: Number of last history Results
    :param shift: Shift to the past/how many recent Results skip. Has to be positive.
    :return: State - Tensor of int/float of shape nLast * (nTasks + 1)
    """
    one_student = results.get(idStudent, [])
    student_results = one_student[-(nLast + shift):(len(one_student) - shift)]
    student_results = [(res.mark, res.task.id) for res in student_results]
    # print("Results: ",student_results)
    return _get_state_inverse(
        tf.reshape(tf.convert_to_tensor(student_results, dtype=tf.float32), shape=(len(student_results), 2)),
        tf.constant(nLast, dtype=tf.int32), tf.constant(nTasks, dtype=tf.int32))


def get_state() -> tf.Variable:
    # if cl_state is None:
    #     __state = tf.Variable([], dtype=tf.float32, trainable=False, shape=(None,))
    return __state


@tf.function(
    input_signature=[tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
                     tf.TensorSpec(shape=(), dtype=tf.int32),
                     tf.TensorSpec(shape=(), dtype=tf.int32)]
)
def _get_state_inverse(student_results: tf.Tensor, nLast: tf.Tensor, nTasks: tf.Tensor) -> tf.Tensor:
    state = get_state()
    state.assign(tf.convert_to_tensor([]))  # initialize empty
    for result in student_results:
        state.assign(tf.concat([state.value(),  # prev state
                                tf.reshape(result[0], shape=(1,)),  # mark
                               # reversed one hot encoding of task ID
                                tf.reverse(tf.one_hot(tf.cast(result[1], tf.int32), nTasks, dtype=tf.float32), [0])],
                               0))
    state.assign(tf.reverse(state, [0]))  # Change positions to ensure order where first is the most recent one
    _complete_empty(state, nLast, nTasks)
    return tf.reshape(tf.convert_to_tensor(state), (1, nLast * (nTasks + 1)))


def get_state_normal(results: Dict, idStudent: int, nLast: int, nTasks: int, shift: int = 0) -> tf.Tensor:
    """
    Function for getting state out of history (self.results) for specified student (in order ... t-2 t-1).
    :param results: Dictionary with list of Results assigned to (int) Student ID
    :param idStudent: Student ID
    :param nTasks: Number of Tasks (during learning process)
    :param nLast: Number of last history Results
    :param shift: Shift to the past/how many recent Results skip. Has to be positive.
    :return: State - Tensor of int/float of shape nLast * (nTasks + 1)
    """
    one_student = results.get(idStudent, [])
    student_results = one_student[-(nLast + shift):(len(one_student) - shift)]
    student_results = [(res.mark, res.task.id) for res in student_results]
    # print("Results: ",student_results)
    return _get_state_normal(
        tf.reshape(tf.convert_to_tensor(student_results, dtype=tf.float32), shape=(len(student_results), 2)),
        tf.constant(nLast, dtype=tf.int32), tf.constant(nTasks, dtype=tf.int32))


@tf.function(
    input_signature=[tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
                     tf.TensorSpec(shape=(), dtype=tf.int32),
                     tf.TensorSpec(shape=(), dtype=tf.int32)]
)
def _get_state_normal(student_results: tf.Tensor, nLast: tf.Tensor, nTasks: tf.Tensor) -> tf.Tensor:
    state = get_state()
    state.assign(tf.convert_to_tensor([]))  # initialize empty
    for result in student_results:
        state.assign(tf.concat([state.value(),  # prev state
                                # one hot encoding of task ID
                                tf.one_hot(tf.cast(result[1], tf.int32), nTasks, dtype=tf.float32),
                                tf.reshape(result[0], shape=(1,))],  # mark
                               0))
    _complete_empty(state, nLast, nTasks)
    return tf.reshape(tf.convert_to_tensor(state), (1, nLast * (nTasks + 1)))
