from abc import ABC, abstractmethod
from typing import List

from . import TeacherRL
from .. import Task
from .state_representation import get_state_inverse, get_state_normal
 
class TeacherNLastHistory(TeacherRL, ABC):
    def __init__(self,
                nSkills: int,
                tasks: List[Task],
                nLast: int,
                inverse_state: bool = True,
                **kwargs) -> None:
        """
        :param nSkills: Number of skills to learn
        :param nLast: Quantity of last Results used in state
        :param tasks: List of Tasks
        :param inverse_state: Bool, should state be in order: t-1, t-2, ...?
        """
        super().__init__(nSkills,tasks,**kwargs)
        self.results = dict()
        self.nLast = nLast
        self.nTasks = len(tasks)
        if inverse_state:
            self._get_state = get_state_inverse
        else:
            self._get_state = get_state_normal
        self.iteration_st = 0
        
    def get_state(self, student, shift=0):
        return self._get_state(self.results, student, self.nLast, self.nTasks, shift)

    def choose_task(self, student):
        return super().choose_task(student.id)
    
    def receive_result(self, result, reward=None, last=False) -> None:
        student = result.idStudent
        if reward is None and not result.isExam:
            self.results[student] = self.results.get(result.idStudent, [])
            self.results[student].append(result)
        if not result.isExam and not last:
            self._receive_result_one_step(result, student, reward, last)
            if reward is not None:
                self.results[student] = []  # remove student history after exam
        
        super().receive_result(result)

  
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
        self.learn(self.get_state(student, shift=1), self.results[student][-1].task.id, self.get_state(student),
                    _reward, done)

    """
    Eryk miał randomowe akcje przez 100 epok -- nie mozna tego zamodelowac na tę chwilę
    """
    # def _receive_results_after_exam(self):
    #     """
    #     Check TeacherNLastHistory
    #     """
    #     self.iteration_st += 1
    #     if self.start_of_random_iteration < self.iteration_st < (
    #             self.start_of_random_iteration + self.number_of_random_iteration):
    #         self.epsilon -= self.epsilon_diff

