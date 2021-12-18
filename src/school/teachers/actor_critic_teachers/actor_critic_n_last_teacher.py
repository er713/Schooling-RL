import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf
from random import choice
from typing import List

from . import _learn_main
from .. import TeacherNLastHistory
from ... import Task
from ..models.actor_critic import *


class ActorCriticNLastTeacher(TeacherNLastHistory):
    def __init__(self,
                 nSkills: int,
                 tasks: List[Task],
                 nLast: int,
                 cnn: bool = False,
                 *args, **kwargs):
        """
        :param nSkills:Number of skills to learn
        :param tasks: List of Tasks
        :param gamma: Basic RL gamma
        :param nLast: Quantity of last Results used in state
        :param learning_rate: Basic ML learning rate
        :param cnn: Bool, should first layer be convolution?
        :param verbose: Bool, if True, prints a lot of states and actions
        :param epsilon: Variable, how many action should be random. Accepted vales <0, 100>!
        :param start_of_random_iteration: From which iteration epsilon should decrease
        :param number_of_random_iteration: How many iteration epsilon should decrease
        :param end_epsilon: To what values epsilon should decrease
        :param nStudents: Number of Students during one exam
        """
        super().__init__(nSkills, tasks, nLast, **kwargs)
        if not cnn:
            self.actor = Actor(self.nTasks, verbose=self.verbose)
            self.critic = Critic()
        else:
            self.actor = ActorCNN(self.nTasks, verbose=self.verbose, nLast=nLast)
            self.critic = CriticCNN(nTasks=self.nTasks, nLast=nLast)
        self.actor_opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.critic_opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def get_action(self, state):
        logits = self.actor(state)
        action_probabilities = tfp.distributions.Categorical(logits=logits)
        action = action_probabilities.sample(sample_shape=())

        # if self.verbose:
        #     print(action)

        action = [task_ for task_ in self.tasks if task_.id == action][0]
        return action

    def learn(self, state: List[int], action: int, next_state: List[int], reward: int, done: int) -> None:
        _learn_main(self.actor, self.critic, state, tf.constant(action), next_state,
                    tf.constant(reward, dtype=tf.float32),
                    tf.constant(done, dtype=tf.float32), tf.constant(self.gamma, dtype=tf.float32), self.actor_opt,
                    self.critic_opt)
