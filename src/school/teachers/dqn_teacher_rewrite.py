from .state_representation import TableTeacher
from typing import List
from ..task import Task
from .models.dqn import *
from .losses import dqn_loss
import tensorflow as tf
import tensorflow_probability as tfp
from random import shuffle
from .. import Result
from copy import deepcopy
from .utils.dqn_structs import *
from . import TeacherNLastHistory
from random import random, randint
from .models.actor_critic import *
from . import losses



class DQNTeacher(TableTeacher):
    """Deep Q-learning agent."""

    def __init__(self, nSkills: int, tasks: List[Task], mem_size=4096, batch_size=1024,
                 **kwargs):
        """Set parameters, initialize network."""
        super().__init__(nSkills, tasks, **kwargs)
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.lossFun = dqn_loss
        modelInputSize = len(tasks) * 2
        self.estimator = DQN(modelInputSize)
        self.targetEstimator = DQN(modelInputSize)
        self.mem = ReplayBuffer(1, self.mem_size, self.batch_size)
        self.noTargetIte = nSkills*len(tasks)*self.time_to_exam
        self.__targetCounter = 0


    def get_action(self, state):
        task_id = tf.argmax(self.estimator(state)[0]).numpy()
        for task in self.tasks:
            if task.id == task_id:
                return task


    def receive_result(self, result, last=False, reward=None) -> None:
        # Exam results need to be reduced in receive_exam_res
        super().receive_result(result, last, reward)
        # copy estimator weights to target estimator after noTargetIte iterations

        self.__update_target()

        if len(self.mem) > self.batch_size:
            self.learn()

    def learn(self):
        states, actions, rewards, next_states, dones = self.mem.sample()
        real_q = []
        states_buff = []
        actions_buff = []

        for i, (r, d, ns, a, s) in enumerate(zip(rewards, dones, next_states, actions, states)):
            if d:
                real_q.append(tf.constant([r], dtype=tf.float32))
            else:
                next_state = tf.expand_dims(tf.constant(ns), axis=0)
                real_q.append(r + self.gamma * self.__get_target_q(next_state)[0])
            states_buff.append(s)
            actions_buff.append((i, a))
        self.estimator.train_step(tf.constant(states_buff), tf.constant(actions_buff), tf.stack(real_q))

    def update_memory(self, result: Result, last: bool):
        # last exam task
        if last and result.isExam:
            exam_results = self.exam_results[result.idStudent]
            state = exam_results.state
            action = exam_results.action
            reward = exam_results.marksSum / exam_results.noAcquiredExamResults
        # non exam task
        else:
            state = self.students.get(result.idStudent, self.create_new_student())
            action = result.task.id
            reward = 0
        # last non exam task, prepare structure to receive exam results
        if last and not result.isExam:
            self.exam_results[result.idStudent] = ExamResultsRecord(state, action)
            return

        # update student state with action (given task id) and result of that action
        student_new_state = self.update_student_state(result)
        self.mem.add(state, action, reward, student_new_state, result.isExam)

    @tf.function
    def __get_target_q(self, state: tf.Tensor):
        modelOutput = self.estimator(state)
        bestActionIdx = tf.expand_dims(tf.argmax(modelOutput, axis=1), axis=1)
        targetQVector = self.targetEstimator(state)
        targetQ = tf.gather(targetQVector, bestActionIdx, axis=1, batch_dims=1)
        return targetQ


    def __update_target(self):
        """
                Copy weights from online network to target networks
                after being called noTargetIte times.
        """
        self.__targetCounter += 1
        if self.__targetCounter == self.noTargetIte:
            self.__targetCounter = 0
            self.targetEstimator.copy_weights(self.estimator)


class DQNTeacherNLastHistory(TeacherNLastHistory):
    
    def __init__(self, nSkills: int, tasks: List[Task], nLast: int, nStudents: int, mem_size=1024, batch_size=64, cnn=False, verbose=False,
                 **kwargs):
        """Set parameters, initialize network."""
        super().__init__(nSkills, tasks, nLast, **kwargs)
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.lossFun = dqn_loss
        if not cnn:
            self.estimator = Actor(self.nTasks, verbose=verbose)
            self.targetEstimator = Actor(self.nTasks, verbose=verbose)
        else:
            self.estimator = ActorCNN(self.nTasks, verbose=verbose, nLast=nLast)
            self.targetEstimator = ActorCNN(self.nTasks, verbose=verbose, nLast=nLast)
        self.estimator_opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        # self.estimator = DQN(modelInputSize)
        # self.targetEstimator = DQN(modelInputSize)
        self.mem = ReplayBuffer(1, self.mem_size, self.batch_size)
        self.noTargetIte = nSkills*len(tasks)
        self.__targetCounter = 0


    def get_action(self, state):
        logits = self.estimator(state)
        action_probabilities = tfp.distributions.Categorical(logits=logits)
        action = action_probabilities.sample(sample_shape=())
        # print(action)
        action = [task_ for task_ in self.tasks if task_.id == action][0]
        return action
 


    def receive_result(self, result, last=False, reward=None) -> None:
        # Exam results need to be reduced in receive_exam_res
        super().receive_result(result, last=last, reward=reward)
        # copy estimator weights to target estimator after noTargetIte iterations
        

    def learn(self):
        states, actions, rewards, next_states, dones = self.mem.sample()
        

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
        self.mem.add(self.get_state(student,shift=1),  self.results[student][-1].task.id, reward_, self.get_state(student,shift=1), done)
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
            q = self.estimator(state)
            q_next = self.estimator(next_state)
            logits = self.targetEstimator(state)

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
            self.targetEstimator.copy_weights(self.estimator)
