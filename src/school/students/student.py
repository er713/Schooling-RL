from abc import abstractmethod


class Student:
    def __init__(self, id: int = 0, proficiency: list[float] = [], desireToLearn: float = 1) -> None:
        """
        :param id: The id of student
        :param proficiency: The list of skills proficiency in range [-1,1].
        :param desireToLearn: The likelihood to do task [0,1]
        """
        self.id = id
        self._proficiency = proficiency
        self._desireToLearn = desireToLearn

    @abstractmethod
    def solve_task(self, task: object) -> None:
        """
        Function resposible for solve task which triggers update proficiency
        :param task: The Task object
        """
        raise NotImplementedError('solv_task was not implemented')

    def want_task(self) -> bool:
        """
        Function which return student choice to solve or not to solve a task,
        """
        raise NotImplementedError('want_task was not implemented')

    @abstractmethod
    def _update_proficiency(self, result: object) -> None:
        """
        Function resposible for update student proficiency of given skill
        :param result: { mark: float, duration: float, task: Task }
        """
        raise NotImplementedError('_update_proficiency was not implemented')
