from .teacher import Teacher
import random
import tensorflow as tf
from typing import List
from abc import abstractmethod


class TeacherRL(Teacher):
    def __init__(self, nSkills: int, tasks: List[Task], epsilon: float = .995, decay_epsilon: float = .999,
    min_eps: float = .01, gamma: float = .8, learning_rate: float = 1e-5 ,**kwargs):
        super.__init__(nSkills, tasks, **kwargs)
        self.epsilon = epsilon
        self.decay_epsilon = decay_epsilon
        self.min_eps = min_eps
        self.gamma = gamma
        self.learning_rate = learning_rate
    
    def choose_task(self, student, is_learning: bool = True) -> Task:
        #ekspoalatacja
        if not is_learning or random() > self.epsilon:
            state = self.get_state(student)
            action = self.get_action(state)
        #eksploracja
        else:
            action = random.choice(self.tasks)
        return action
    
    def receive_result(self, result, reward=None, last=False) -> None:
        self.epsilon = max(self.min_eps, self.epsilon - self.decay_epsilon)
    
    @abstractmethod
    def get_state(self):
        raise NotImplementedError('get_state was not implemented')

    @abstractmethod
    def get_action(self, state):
        raise NotImplementedError('get_action was not implemented')

    @abstractmethod
    def learn(self, state, action, next_state, reward,done):
        raise NotImplementedError('learn was not implemented')

    @abstractmethod
    def state_handler(self, input):
        raise NotImplementedError('state_handler was not implemented')

