from typing import List
from . import TeacherNLastHistory
from .. import Task
from .layers import EmbeddedTasks
from abc import ABC
from .state_representation import get_state_history, get_state_for_rnn


class TeacherAllHistory(TeacherNLastHistory, ABC):
    def __init__(self, nSkills: int, tasks: List[Task], task_embedding_size: int = 5, base_history: int = 5, **kwargs):
        super().__init__(nSkills, tasks, None, None, **kwargs)

        self.task_embedding_size = task_embedding_size
        self.base_history = base_history
        self.embedding_for_tasks = EmbeddedTasks(self.nTasks, self.task_embedding_size, self.base_history)

    def get_state(self, student, shift=0):
        return get_state_history(self.results, student, shift)


class TeacherAllHistoryRNN(TeacherNLastHistory, ABC):
    def __init__(self, nSkills: int, tasks: List[Task], task_embedding_size: int = 5, rnn_units: int = 5, **kwargs):
        del kwargs['nLast']
        super().__init__(nSkills, tasks, None, None, **kwargs)

        self.task_embedding_size = task_embedding_size
        self.rnn_units = rnn_units
        self.last_rnn_state = None

    def get_state(self, student, shift=0):
        return get_state_for_rnn(self.results.get(student, [[None, None], [None, None]])[-shift])

    def receive_result(self, result, reward=None, last=False) -> None:
        student = result.idStudent
        if reward is None and not result.isExam:
            self.results[student] = self.results.get(student, [[None, None], [None, None]])
            self.results[student] = [self.results[student][1], [None, None]]
            self.results[student][1][1] = self.last_rnn_state
            self.results[student][1][0] = result
            self.last_rnn_state = None
        if not result.isExam and not last:
            self._receive_result_one_step(result, student, reward, last)
            if reward is not None:
                self.results[student] = [[None, None], [None, None]]  # remove student history after exam
                super(TeacherNLastHistory, self).receive_result(result)

    def _receive_result_one_step(self, result, student, reward=None, last=False) -> None:
        """
        Check TeacherNLastHistory
        """
        if reward is None:
            done = 0
            _reward = 0
        else:
            done = 1
            _reward = reward
            self.iteration_st += 1
        self.learn(self.get_state(student, shift=1), self.results[student][1][0].task.id, self.get_state(student),
                   _reward, done)
