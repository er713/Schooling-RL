from typing import List
import tensorflow_probability as tfp
import tensorflow as tf
from random import choice
import numpy as np

from . import losses
from . import TeacherNLastHistory
from .. import Task
from .models.actor_critic import *


class TeacherActorCritic(TeacherNLastHistory):
    def __init__(self, nSkills: int, tasks: List[Task], gamma: float, nLast: int, learning_rate: int, nStudents: int,
                 cnn: bool = False, verbose: bool = False, epsilon: float = 90, start_of_random_iteration=50,
                 number_of_random_iteration=50, end_epsilon=0, *args, **kwargs):
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
        super().__init__(nSkills, nLast, tasks, **kwargs)

        if not cnn:
            self.actor = Actor(self.nTasks, verbose=verbose)
            self.critic = Critic()
        else:
            self.actor = ActorCNN(self.nTasks, verbose=verbose, nLast=nLast)
            self.critic = CriticCNN(nTasks=self.nTasks, nLast=nLast)
        self.actor_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.critic_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.gamma = gamma
        self.verbose = verbose
        self.epsilon = epsilon

        # self.random_action = True
        self.iteration_st = 0  # Number of student who finished exam
        self.start_of_random_iteration = start_of_random_iteration * nStudents
        self.number_of_random_iteration = number_of_random_iteration * nStudents
        self.epsilon_diff = (epsilon - end_epsilon) / number_of_random_iteration
        self.end_epsilon = end_epsilon

        self.choices = np.zeros((len(self.tasks),), dtype=np.int_)  # Only for checking if action is diverse

    def choose_task(self, student) -> Task:
        if choice(range(100)) < self.epsilon:  # Random action
            action = choice(range(len(self.tasks)))
        else:  # Actor based action
            state = self.get_state(student.id)
            logits = self.actor(state)
            action_probabilities = tfp.distributions.Categorical(logits=logits)
            action = action_probabilities.sample(sample_shape=())
            self.choices[action.numpy()[0]] += 1
        if self.verbose:
            print(action)
        task = [task_ for task_ in self.tasks if task_.id == action][0]
        return task

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
        self._learn(self.get_state(student, shift=1), self.results[student][-1].task.id, self.get_state(student),
                    _reward, done)

    def _receive_results_after_exam(self):
        """
        Check TeacherNLastHistory
        """
        self.iteration_st += 1
        if self.start_of_random_iteration < self.iteration_st < (
                self.start_of_random_iteration + self.number_of_random_iteration):
            self.epsilon -= self.epsilon_diff

    """
    Mając state wykonaj akcje a, zaobserwuj nagrodę reward i następnik next_state
    """

    def _learn(self, state: List[int], action: int, next_state: List[int], reward: int, done: int) -> None:
        _learn_main(self.actor, self.critic, state, action, next_state, reward, done, self.gamma, self.actor_opt,
                    self.critic_opt)


@tf.function
def _learn_main(actor: tf.keras.Model, critic: tf.keras.Model, state: List[int], action: int, next_state: List[int],
                reward: int, done: int, gamma: float, actor_opt, critic_opt) -> None:
    """
    Dokumentacja
    """
    with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
        q = critic(state)
        q_next = critic(next_state)
        logits = actor(state)

        # δ = reward + self.gamma * q_next * (1 - done) - q  # this works w/o tf.function
        δ = float(reward) + float(gamma * q_next * (1 - done)) - float(q)  # float only for tf.function

        actor_loss = losses.actor_loss(logits, action, δ)
        critic_loss = δ ** 2  # MSE?

    actor_grads = actor_tape.gradient(actor_loss, actor.trainable_variables)
    critic_grads = critic_tape.gradient(critic_loss, critic.trainable_variables)

    actor_opt.apply_gradients(zip(actor_grads, actor.trainable_variables))
    critic_opt.apply_gradients(zip(critic_grads, critic.trainable_variables))
