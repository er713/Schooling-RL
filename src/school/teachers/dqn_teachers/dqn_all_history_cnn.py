from typing import List
from ...task import Task
from ..losses import dqn_loss
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from ..utils.dqn_structs import *
from .. import TeacherAllHistory
from ..models.actor_critic import *
from .. import losses
from ..layers import AllHistoryCNN, EmbeddedTasks


class DQNTeacherAllHistoryCNN(TeacherAllHistory):

    def __init__(self, nSkills: int, tasks: List[Task], nStudents: int, mem_size=1024, batch_size=64,
                 cnn=False, verbose=False, filters: int = 5, task_embedding_size: int = 5, base_history: int = 5,
                 **kwargs):
        """Set parameters, initialize network."""
        super().__init__(nSkills, tasks, task_embedding_size, base_history, **kwargs)
        self.nStudents = nStudents
        self.mem_size = mem_size
        self.batch_size = batch_size
        self._estimator = Actor(self.nTasks, verbose=verbose)
        self._targetEstimator = Actor(self.nTasks, verbose=verbose)
        self.estimator = tf.keras.Sequential([
            self.embedding_for_tasks,
            AllHistoryCNN(self.task_embedding_size, self.base_history, filters),
            self._estimator])
        self.targetEstimator = tf.keras.Sequential([
            EmbeddedTasks(self.nTasks, self.task_embedding_size, self.base_history),
            AllHistoryCNN(self.task_embedding_size, self.base_history, filters),
            self._targetEstimator
        ])
        self.estimator_opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        # self.estimator = DQN(modelInputSize)
        # self.targetEstimator = DQN(modelInputSize)
        self.mem = ReplayBuffer(1, self.mem_size, self.batch_size)
        self.noTargetIte = nSkills * len(tasks)
        self.__targetCounter = 0
        self.verbose = verbose
        self.choices = np.zeros((self.nTasks,), dtype=np.int_)

    def get_action(self, state):
        logits = self.estimator(state)
        action_probabilities = tfp.distributions.Categorical(logits=logits)
        action = action_probabilities.sample(sample_shape=())
        self.choices[action[0]] += 1
        for task in self.tasks:
            if task.id == action:
                return task

    def receive_result(self, result, last=False, reward=None) -> None:
        # Exam results need to be reduced in receive_exam_res
        super().receive_result(result, last=last, reward=reward)
        # copy estimator weights to target estimator after noTargetIte iterations

    def learn(self):
        states, actions, rewards, next_states, dones = self.mem.sample()
        # reward = 1000*reward + 0.001
        if self.verbose:
            self.iteration_st += 1
            if self.iteration_st % (self.nStudents * 10) == 0:
                print(self.iteration_st)
                print('epsilon:', self.epsilon)
                print('variety:\n',
                      np.reshape(self.choices, (self.choices.shape[0] // 7, 7)))
                self.choices = np.zeros((len(self.tasks),), dtype=np.int_)
        for i, (r, d, ns, a, s) in enumerate(zip(rewards, dones, next_states, actions, states)):
            self._learn_main(state=tf.constant(s, dtype=tf.float32), action=tf.constant(a, dtype=tf.float32),
                             next_state=tf.constant(ns, dtype=tf.float32), reward=tf.constant(r, dtype=tf.float32),
                             done=tf.constant(d, dtype=tf.float32))

    def _receive_result_one_step(self, result, student, reward=None, last=False) -> None:
        if reward is None:
            done = 0
            reward_ = 0
        else:
            done = 1
            reward_ = reward
        # update student state with action (given task id) and result of that action
        self.mem.add(self.get_state(student, shift=1), self.results[student][-1].task.id, reward_,
                     self.get_state(student, shift=0), done)
        if len(self.mem) > self.batch_size:
            self.learn()
            self.__update_target()

    @tf.function
    def _learn_main(self, state: tf.Tensor, action: tf.Tensor, next_state: tf.Tensor,
                    reward: tf.Tensor, done: tf.Tensor) -> None:
        """
        Dokumentacja
        """
        with tf.GradientTape() as estimator_tape:
            q = self.targetEstimator(state)
            q_next = self.targetEstimator(next_state)
            logits = self.estimator(state)

            δ = reward + self.gamma * q_next * (1 - done) - q  # this works w/o tf.function
            # δ = float(reward) + float(gamma * q_next * (1 - done)) - float(q)  # float only for tf.function

            estimator_loss = losses.actor_loss(logits, action, δ)

        estimator_grads = estimator_tape.gradient(estimator_loss, self.estimator.trainable_variables)

        self.estimator_opt.apply_gradients(zip(estimator_grads, self.estimator.trainable_variables))

    def __update_target(self):
        """
                Copy weights from online network to target networks
                after being called noTargetIte times.
        """
        self.__targetCounter += 1
        if self.__targetCounter == self.noTargetIte:
            self.__targetCounter = 0
            # self.targetEstimator.copy_weights(self.estimator)
            self.targetEstimator.set_weights(self.estimator.get_weights())
