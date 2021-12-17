from typing import List, Dict
import tensorflow as tf


def get_state_inverse(results: Dict, idStudent: int, shift: int = 0) -> tf.Tensor:
    """
    Function for getting state out of history (results) for specified student (in order t-1, t-2, ...).
    :param results: Dictionary with list of Results assigned to (int) Student ID
    :param idStudent: Student ID
    :param shift: Shift to the past/how many recent Results skip. Has to be positive.
    :return: State - Tensor of int/float of shape nLast * (nTasks + 1)
    """
    one_student = results.get(idStudent, [])
    student_results = one_student[:(-shift)]
    student_results = [(res.task.id, res.mark) for res in student_results]
    # print("Results: ",student_results)
    return _get_state_inverse(
        tf.reshape(tf.convert_to_tensor(student_results, dtype=tf.float32), shape=(len(student_results), 2)))


state = tf.Variable([[], []], dtype=tf.float32, trainable=False, shape=(2, None))


@tf.function(input_signature=[tf.TensorSpec(shape=(None, 2), dtype=tf.float32)])
def _get_state_inverse(student_results: tf.Tensor) -> tf.Tensor:
    global state
    state.assign(tf.convert_to_tensor([[], []]))  # initialize empty
    for result in student_results:
        state.assign(tf.concat([state.value(),  # prev state
                                tf.reshape(result, shape=(2, 1))], 1))
    # state.assign(tf.reverse(state, [1]))  # Change positions to ensure order where first is the most recent one
    # _complete_empty(state, nLast, nTasks)
    return tf.convert_to_tensor(state)


def get_state_for_rnn(results: List):
    if results[0] is not None:
        state_task = _get_state_inverse(
            tf.convert_to_tensor([[results[0].task.id, results[0].mark]], dtype=tf.float32)
        )
    else:
        state_task = tf.convert_to_tensor([[], []], dtype=tf.float32)
    return state_task, results[1]
