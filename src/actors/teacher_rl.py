from abc import abstractmethod
from random import random, choice
from typing import List

from actors.const import EPSILON, DECAY_EPSILON, MIN_EPS, GAMMA, LEARNING_RATE
from actors.teacher import Teacher
from environment.task import Task


class TeacherRL(Teacher):
    def __init__(
        self,
        nSkills: int,
        tasks: List[Task],
        epsilon: float = EPSILON,
        decay_epsilon: float = DECAY_EPSILON,
        min_eps: float = MIN_EPS,
        gamma: float = GAMMA,
        learning_rate: float = LEARNING_RATE,
        **kwargs
    ):
        super().__init__(nSkills, tasks, **kwargs)
        self.epsilon = epsilon
        self.decay_epsilon = decay_epsilon
        self.min_eps = min_eps
        self.gamma = gamma
        self.learning_rate = learning_rate

    def choose_task(self, student, is_learning: bool = True) -> Task:
        # ekspoalatacja
        if not is_learning or random() > self.epsilon:
            state = self.get_state(student)
            action = self.get_action(state)
        # eksploracja
        else:
            action = choice(self.tasks)
        return action

    def receive_result(self, result, reward=None, last=False) -> None:
        self.epsilon = max(self.min_eps, self.epsilon * self.decay_epsilon)

    @abstractmethod
    def get_state(self, student):
        raise NotImplementedError("get_state was not implemented")

    @abstractmethod
    def get_action(self, state):
        raise NotImplementedError("get_action was not implemented")

    @abstractmethod
    def learn(self, state, action, next_state, reward, done):
        raise NotImplementedError("learn was not implemented")

    @abstractmethod
    def state_handler(self, input):
        raise NotImplementedError("state_handler was not implemented")
