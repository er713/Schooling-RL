from abc import abstractmethod
from typing import List
from .. import Task


class Teacher:
    def __init__(self, nSkills: int, tasks: List[Task], **kwargs) -> None:
        self.tasks = tasks
        self.nSkills = nSkills

    @abstractmethod
    def choose_task(self, student) -> Task:
        """
        Function responsible for choosing task for given student
        :param student: student who want a task
        :return: return Task for student
        """
        raise NotImplementedError('choose_task was not implemented')

    @abstractmethod
    def receive_result(self, result, reward=None, last=False) -> None:
        raise NotImplementedError('receive_result was not implemented')

    # @abstractmethod
    def __str__(self):
        return self.__class__.__name__
