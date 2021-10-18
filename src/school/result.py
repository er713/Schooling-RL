from __future__ import annotations
from typing import List
from numpy import mean


class Result:
    def __init__(self, mark: float, duration: float, task: object, idStudent: int) -> None:
        """
        :param mark: result of task (for now 0 or 1 but in future it can be continuous [0,1])
        :param duration: how long it take to solve the task 
        :param task: task which has been done
        :param idStudent: id of student who done the task 
        """
        self.mark = mark
        self.duration = duration
        self.task = task
        self.idStudent = idStudent

    @staticmethod
    def get_mean_result(results: List[Result]) -> (float, float):
        """
        Calculate mean score
        :param results: List of results
        :return: (mean score, mean time)
        """
        return mean([result.mark for result in results]), mean([result.duration for result in results])
