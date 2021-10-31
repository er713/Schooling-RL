from abc import abstractmethod
from typing import List

from .. import Task, Result


class Student:
    def __init__(self, id: int = 0, proficiency: List[float] = [], desireToLearn: float = 1) -> None:
        """
        :param id: The id of student
        :param proficiency: The list of skills proficiency in range [-1,1].
        :param desireToLearn: The likelihood to do task [0,1]
        """
        self.id = id
        self._proficiency = proficiency
        self._desireToLearn = desireToLearn

    @abstractmethod
    def solve_task(self, task: Task, isExam: bool = False) -> Result:
        """
        Function responsible for solve task which triggers update proficiency
        :param task: The Task object
        :param isExam: Does task is part of the exam.
        """
        raise NotImplementedError('solv_task was not implemented')

    @abstractmethod
    def want_task(self) -> bool:
        """
        Function which return student choice to solve or not to solve a task,
        """
        raise NotImplementedError('want_task was not implemented')

    @abstractmethod
    def _update_proficiency(self, result: Result) -> None:
        """
        Function responsible for update student proficiency of given skill
        :param result: { mark: float, duration: float, task: Task }
        """
        raise NotImplementedError('_update_proficiency was not implemented')

    def __str__(self):
        return "studentID: " + str(self.id) + \
               "\ndesireToLearn: " + str(self._desireToLearn) + \
               "\nprofficiencies: "+ str(self._proficiency)


