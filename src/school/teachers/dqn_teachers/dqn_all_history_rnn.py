from typing import List
from ...task import Task
from ..losses import dqn_loss
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from ..utils.dqn_structs import *
from .. import TeacherAllHistoryRNN
from ..models.actor_critic import *
from .. import losses
from ..models import RNNWrapper
from ..constants import BATCH_SIZE, MEM_SIZE, TARGET_ITER, LEARN



class DQNTeacherAllHistoryRNN(TeacherAllHistoryRNN):

    def __init__(self, nSkills: int, tasks: List[Task], mem_size=MEM_SIZE, batch_size=BATCH_SIZE,
                 cnn=False, verbose=False, task_embedding_size: int = None, rnn_units: int = None,
                 **kwargs):
        """Set parameters, initialize network."""
        if rnn_units is None:
            rnn_units = max(3 * nSkills, 5)
        if task_embedding_size is None:
            task_embedding_size = np.ceil(len(tasks) * 0.6)
        super().__init__(nSkills, tasks, task_embedding_size, rnn_units, **kwargs)
        # self.nStudents = nStudents
        self.mem_size = mem_size
        self.batch_size = batch_size

        self._estimator = Actor(self.nTasks, verbose=verbose)
        self._targetEstimator = Actor(self.nTasks, verbose=verbose)
        self.estimator = RNNWrapper(self.rnn_units, self.nTasks, task_embedding_size, [self._estimator])
        self.targetEstimator = RNNWrapper(self.rnn_units, self.nTasks, task_embedding_size, [self._targetEstimator])
        self.estimator_opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.mem = ReplayBuffer(1, self.mem_size, self.batch_size)
        self.noTargetIte = TARGET_ITER
        self.__targetCounter = 0

        self.verbose = verbose
        self.choices = np.zeros((self.nTasks,), dtype=np.int_)
        self.__learnCounter = 0

    def get_action(self, state):
        logits, st = self.estimator.get_specific_call(state[0], state[1])
        self.last_rnn_state = st
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
        if LEARN == self.__learnCounter:
            self.__learnCounter = 0
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
                self._learn_main(state=s, action=a,
                                next_state=ns, reward=tf.constant(r, dtype=tf.float32),
                                done=tf.constant(d, dtype=tf.float32))
        self.__learnCounter += 1

    def _receive_result_one_step(self, result, student, reward=None, last=False) -> None:
        if reward is None:
            done = 0
            reward_ = 0
        else:
            done = 1
            reward_ = reward
        # update student state with action (given task id) and result of that action
        self.mem.add(self.get_state(student, shift=1), self.results[student][1][0].task.id, reward_,
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
            q, st = self.targetEstimator.get_specific_call(state[0], state[1])
            q_next, _ = self.targetEstimator.get_specific_call(next_state[0], next_state[1])
            logits, _ = self.estimator.get_specific_call(state[0], state[1])

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
