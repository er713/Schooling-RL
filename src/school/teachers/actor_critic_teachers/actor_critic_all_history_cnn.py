import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf
from typing import List

from . import _learn_main
from .. import TeacherAllHistory
from ..layers import AllHistoryCNN
from ... import Task
from ..models.actor_critic import *


class ActorCriticAllHistoryTeacher(TeacherAllHistory):
    def __init__(self,
                 nSkills: int,
                 tasks: List[Task],
                 nStudents: int,
                 cnn: bool = False,
                 verbose: bool = False,
                 task_embedding_size: int = 10,
                 base_history: int = 7,
                 filters: int = 5,
                 *args, **kwargs):
        super().__init__(nSkills, tasks, task_embedding_size, base_history, **kwargs)

        self._actor = Actor(self.nTasks)
        self._critic = Critic()
        if cnn:
            self.actor = tf.keras.Sequential(
                [self.embedding_for_tasks,
                 AllHistoryCNN(self.task_embedding_size, self.base_history, filters),
                 self._actor])
            self.critic = tf.keras.Sequential(
                [self.embedding_for_tasks,
                 AllHistoryCNN(self.task_embedding_size, self.base_history, filters),
                 self._critic])
        self.actor_opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.critic_opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.nStudents = nStudents
        self.verbose = verbose
        # if verbose:
        self.choices = np.zeros((self.nTasks,), dtype=np.int_)

    def get_action(self, state):
        logits = self.actor(state)
        action_probabilities = tfp.distributions.Categorical(logits=logits)
        action = action_probabilities.sample(sample_shape=())

        # if self.verbose:
        #     print(action.numpy())
        self.choices[action[0]] += 1

        action = [task_ for task_ in self.tasks if task_.id == action.numpy()[0]][0]
        # if self.verbose:
        #     print(action)
        return action

    def learn(self, state: List[int], action: int, next_state: List[int], reward: int, done: int) -> None:
        if self.verbose:
            self.iteration_st += 1
            if self.iteration_st % (self.nStudents*10) == 0:
                print(self.iteration_st)
                print('epsilon:', self.epsilon)
                print('variety:\n',
                      np.reshape(self.choices, (self.choices.shape[0] // 7, 7)))
                self.choices = np.zeros((len(self.tasks),), dtype=np.int_)

        _learn_main(self.actor, self.critic, state, tf.constant(action), next_state,
                    tf.constant(1000 * reward + 0.001, dtype=tf.float32),
                    tf.constant(done, dtype=tf.float32), tf.constant(self.gamma, dtype=tf.float32), self.actor_opt,
                    self.critic_opt)
