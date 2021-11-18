from typing import List
import tensorflow_probability as tfp
import tensorflow as tf
from random import choice
import numpy as np

from . import losses
from . import Teacher
from .. import Task
from .models.actor_critic import *


class TeacherActorCritic(Teacher):
    def __init__(self, nSkills: int, tasks: List[Task], gamma: float, nLast: int, learning_rate: int, cnn: bool = False,
                 verbose: bool = False, epsilon: float = 90, start_of_random_iteration=50,
                 number_of_random_iteration=50, end_epsilon=0, *args, **kwargs):
        super().__init__(nSkills, tasks, **kwargs)
        self.nTasks = len(tasks)
        if not cnn:
            self.actor = Actor(self.nTasks, verbose=verbose)
            self.critic = Critic()
        else:
            self.actor = ActorCNN(self.nTasks, verbose=verbose, nLast=nLast)
            self.critic = CriticCNN(nTasks=self.nTasks, nLast=nLast)
        self.actor_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.critic_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.gamma = gamma
        self.nLast = nLast
        self.verbose = verbose
        self.epsilon = epsilon
        self.results = dict()
        # self.random_action = True
        self.iteration = 0
        self.start_of_random_iteration = start_of_random_iteration
        self.number_of_random_iteration = number_of_random_iteration
        self.end_epsilon = end_epsilon
        self.epsilon_diff = (epsilon - end_epsilon) / number_of_random_iteration

        self.choices = np.zeros((len(self.tasks),), dtype=np.int_)  # Only for checking if action is diverse

    @tf.function
    def _get_state(self, idStudent: int, shift: int = 0) -> tf.Tensor:
        """
        Function for getting state out of history (self.results) for specified student.
        :param idStudent: Student ID
        :param shift: Shift to the past/how many recent Results skip. Has to be positive.
        :return: State - list of int/float of shape 3*self.nLast
        """
        one_student = self.results.get(idStudent, [])
        student_results = one_student[-(self.nLast + shift):(len(one_student) - shift)]
        # print("Results: ",student_results)
        state = []
        for result in student_results:
            state.append(result.mark)
            tmp = [0.] * self.nTasks
            tmp[result.task.id] = 1.
            [state.append(t) for t in tmp[::-1]]
            # state.append(result.task.id)
            # state.append(list(result.task.taskDifficulties.keys())[0])
        state = state[::-1]  # Change positions to ensure order where first is the most recent one
        while len(state) < self.nLast * (self.nTasks + 1):
            [state.append(0.0) for _ in
             range(self.nTasks)]  # TODO: ustalić co będzie pustym elementem/do wypełnienia brakujących wartości
            state.append(0)

        # print("State: ",state)
        return tf.reshape(tf.convert_to_tensor(state), [1, self.nLast * (self.nTasks + 1)])

    def choose_task(self, student) -> Task:
        if choice(range(100)) < self.epsilon:
            action = choice(range(len(self.tasks)))
            # self.random_action = True
        else:
            state = self._get_state(student.id)
            logits = self.actor(state)
            action_probabilities = tfp.distributions.Categorical(logits=logits)
            action = action_probabilities.sample(sample_shape=())
            self.choices[action.numpy()[0]] += 1
            # print(action.numpy()[0])
            # self.random_action = False
        if self.verbose:
            print(action)
        task = [task_ for task_ in self.tasks if task_.id == action][0]
        return task

    def receive_result(self, result, reward=None, last=False) -> None:
        # if not self.random_action:
        student = result.idStudent
        if reward is None and not result.isExam:
            self.results[student] = self.results.get(result.idStudent, [])
            self.results[student].append(result)
        if not result.isExam and not last:
            if reward is None:
                done = 0
            else:
                done = 1
            if reward is None:
                reward = 0
            self._learn(self._get_state(student, shift=1), self.results[student][-1].task.id,
                        self._get_state(student), reward, done)
            if reward > 0:
                self.iteration += 1
                self.results[student] = []  # remove student history after exam
                if self.start_of_random_iteration < self.iteration < (
                        self.start_of_random_iteration + self.number_of_random_iteration):
                    self.epsilon -= self.epsilon_diff

    """
    Mając state wykonaj akcje a, zaobserwuj nagrodę reward i następnik next_state
    """

    @tf.function
    def _learn(self, state: List[int], action: int, next_state: List[int], reward: int, done: int) -> None:
        """
        Dokumentacja
        """
        with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
            q = self.critic(state)
            q_next = self.critic(next_state)
            logits = self.actor(state)

            # δ = reward + self.gamma * q_next * (1 - done) - q  # this works w/o tf.function
            δ = float(reward) + float(self.gamma * q_next * (1 - done)) - float(q)  # float only for tf.function

            actor_loss = losses.actor_loss(logits, action, δ)
            critic_loss = δ ** 2  # MSE?

        actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)

        self.actor_opt.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        self.critic_opt.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
