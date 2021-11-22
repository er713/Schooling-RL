from typing import List, Dict

import tensorflow as tf


def _complete_empty(state: List, nLast: int, nTasks: int) -> List:
    while len(state) < nLast * (nTasks + 1):
        [state.append(0.0) for _ in
         range(nTasks)]  # TODO: ustalić co będzie pustym elementem/do wypełnienia brakujących wartości
        state.append(0)
    return state


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
    # print("Results: ",student_results)
    return _get_state_inverse(student_results, nLast, nTasks)


@tf.function
def _get_state_inverse(student_results: List, nLast: int, nTasks: int) -> tf.Tensor:
    state = []
    for result in student_results:
        state.append(result.mark)
        tmp = [0.] * nTasks
        tmp[result.task.id] = 1.
        [state.append(t) for t in tmp[::-1]]
        # state.append(result.task.id)
        # state.append(list(result.task.taskDifficulties.keys())[0])
    state = state[::-1]  # Change positions to ensure order where first is the most recent one
    state = _complete_empty(state, nLast, nTasks)
    # print("State: ",state)
    return tf.reshape(tf.convert_to_tensor(state), [1, nLast * (nTasks + 1)])


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
    # print("Results: ",student_results)
    return _get_state_normal(student_results, nLast, nTasks)


@tf.function
def _get_state_normal(student_results: List, nLast: int, nTasks: int) -> tf.Tensor:
    state = []
    for result in student_results:
        tmp = [0.] * nTasks
        tmp[result.task.id] = 1.
        [state.append(t) for t in tmp[::-1]]
        state.append(result.mark)
        # state.append(result.task.id)
        # state.append(list(result.task.taskDifficulties.keys())[0])
    state = _complete_empty(state, nLast, nTasks)
    # print("State: ",state)
    return tf.reshape(tf.convert_to_tensor(state), [1, nLast * (nTasks + 1)])
