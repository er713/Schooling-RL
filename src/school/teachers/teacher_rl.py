from .teacher import Teacher
from .. import Task
import random
import tensorflow as tf
from typing import List
from abc import abstractmethod
from .constants import *


class TeacherRL(Teacher):
    def __init__(self, nSkills: int,
                 tasks: List[Task],
                 epsilon: float = EPSILON,
                 decay_epsilon: float = DECAY_EPSILON,
                 min_eps: float = MIN_EPS,
                 gamma: float = GAMMA,
                 learning_rate: float = LEARNING_RATE,
                 **kwargs):
        super().__init__(nSkills, tasks, **kwargs)
        self.epsilon = epsilon
        self.decay_epsilon = decay_epsilon
        self.min_eps = min_eps
        self.gamma = gamma
        self.learning_rate = learning_rate

    def choose_task(self, student, is_learning: bool = True) -> Task:
        # ekspoalatacja
        if not is_learning or random.random() > self.epsilon:
            state = self.get_state(student)
            action = self.get_action(state)
        # eksploracja
        else:
            action = random.choice(self.tasks)
        return action

    def receive_result(self, result, reward=None, last=False) -> None:
        self.epsilon = max(self.min_eps, self.epsilon * self.decay_epsilon)

    @abstractmethod
    def get_state(self, student):
        raise NotImplementedError('get_state was not implemented')

    @abstractmethod
    def get_action(self, state):
        raise NotImplementedError('get_action was not implemented')

    @abstractmethod
    def learn(self, state, action, next_state, reward, done):
        raise NotImplementedError('learn was not implemented')

    @abstractmethod
    def state_handler(self, input):
        raise NotImplementedError('state_handler was not implemented')
