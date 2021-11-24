from abc import ABC, abstractmethod
from typing import List

from . import Teacher
from .. import Task
from .utils import get_state_inverse, get_state_normal


class TeacherNLastHistory(Teacher, ABC):
    def __init__(self, nSkills: int, nLast: int, tasks: List[Task], inverse_state: bool = True, **kwargs):
        """
        :param nSkills: Number of skills to learn
        :param nLast: Quantity of last Results used in state
        :param tasks: List of Tasks
        :param inverse_state: Bool, should state be in order: t-1, t-2, ...?
        """
        super().__init__(nSkills, tasks, **kwargs)
        self.results = dict()
        self.nLast = nLast
        self.nTasks = len(tasks)
        if inverse_state:
            self._get_state = get_state_inverse
        else:
            self._get_state = get_state_normal

    def get_state(self, student, shift=0):
        return self._get_state(self.results, student, self.nLast, self.nTasks, shift)

    def receive_result(self, result, reward=None, last=False) -> None:
        student = result.idStudent
        if reward is None and not result.isExam:
            self.results[student] = self.results.get(result.idStudent, [])
            self.results[student].append(result)
        if not result.isExam and not last:
            self._receive_result_one_step(result, student, reward, last)
            if reward is not None:
                self.results[student] = []  # remove student history after exam
                self._receive_results_after_exam()

    @abstractmethod
    def _receive_result_one_step(self, result, student, reward=None, last=False) -> None:
        """
        Action after one step
        """
        pass

    @abstractmethod
    def _receive_results_after_exam(self) -> None:
        """
        Action when one Student finished exam
        """
        pass
