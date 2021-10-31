from __future__ import annotations
from typing import List, Optional, Dict
from numpy import mean
from copy import copy

from . import Task


class Result:
    def __init__(self, mark: float, duration: float, task: Optional[Task], idStudent: int, isExam: bool) -> None:
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
        self.isExam = isExam

    def get_dict(self) -> dict:
        """
        :return: Dictionary for saving in csv
        """
        d = copy(self.__dict__)
        d['task'] = d['task'].id
        return d

    @staticmethod
    def get_mean_result(results: List[Result]) -> (float, float):
        """
        Calculate mean score and time.
        :param results: List of results.
        :return: (mean score, mean time)
        """
        return mean([result.mark for result in results]), mean([result.duration for result in results])

    @staticmethod
    def get_exams_means(results: List[Result]) -> List[float]:
        exams = [[]]
        prev = False
        for result in results:
            if result.isExam:
                exams[-1].append(result)
                prev = True
            elif prev:
                exams.append([])
                prev = False
        return [float(Result.get_mean_result(exam)[0]) for exam in exams]

    @classmethod
    def create_from_dict(cls, data: Dict[str, str], tasks: List[Task] = None) -> Result:
        result = cls(None, None, None, None, None)
        result.mark = float(data['mark'])
        result.duration = float(data['duration'])
        result.idStudent = int(data['idStudent'])
        result.isExam = data['isExam'] == 'True'
        if type(result.task) is int and tasks is not None:
            result.task = [task for task in tasks if task.id == result.task][0]
        else:
            result.task = None
        return result
