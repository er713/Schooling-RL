from abc import abstractmethod


class Teacher:
    def __init__(self) -> None:
       raise NotImplementedError('Teacher constructor was not implemented')

    @abstractmethod
    def choose_task(self, student) -> object:
        """
        Fucntion resposible for choosing task for given student
        :param student: student who want a task
        :return: return Task for student
        """
        raise NotImplementedError('choose_task was not implemented')
    
    @abstractmethod
    def recive_result(self, result) -> None:
        raise NotImplementedError('recive_result was not implemented')
