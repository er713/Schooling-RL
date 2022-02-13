from collections import namedtuple, deque
from random import sample
from typing import *

import gym
import tensorflow as tf
from gym import Env
import numpy as np
from actors.const import MEM_SIZE, BATCH_SIZE, TARGET_ITER, LEARN
from actors.dqn_model import DQN
from actors.losses import dqn_loss
from actors.table_teacher import TableTeacher
from environment.result import Result


class Record:
    def __init__(self, state: List[float] = None, action: int = None):
        self.state = state
        self.action = action


class ExamResultsRecord(Record):
    def __init__(self, state: List[float] = None, action: int = None):
        super(ExamResultsRecord, self).__init__(state, action)
        self.noAcquiredExamResults = 0
        self.marksSum = 0

    def clear(self):
        self.state = None
        self.action = None
        self.noAcquiredExamResults = 0
        self.marksSum = 0


class MemoryRecord(Record):
    def __init__(
        self,
        state: List[float] = None,
        action: int = None,
        reward: int = None,
        nState: List[float] = None,
        done: bool = None,
    ):
        super(MemoryRecord, self).__init__(state, action)
        self.reward = reward
        self.nextState = nState
        self.done = done


class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = sample(self.memory, k=self.batch_size)

        states = [e.state for e in experiences if e is not None]
        actions = [e.action for e in experiences if e is not None]
        rewards = [e.reward for e in experiences if e is not None]
        next_states = [e.next_state for e in experiences if e is not None]
        dones = [e.done for e in experiences if e is not None]

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


class DQNTableTeacher(TableTeacher):
    """Deep Q-learning agent."""

    def __init__(
        self, env_name: str, mem_size=MEM_SIZE, batch_size=BATCH_SIZE, **kwargs
    ):
        super().__init__(nSkills=-1, tasks=[])
        self.env: Env = gym.make(env_name)
        self.state = self.env.reset()

        self.mem_size = mem_size
        self.batch_size = batch_size
        self.lossFun = dqn_loss

        self.modelInputSize = self.env.observation_space.shape[0]
        self.modelOutputSize = self.env.action_space.n
        self.estimator = DQN(self.modelInputSize, self.modelOutputSize)
        self.targetEstimator = DQN(self.modelInputSize, self.modelOutputSize)

        self.mem = ReplayBuffer(1, self.mem_size, self.batch_size)
        self.noTargetIte = TARGET_ITER

        self.__targetCounter = 0
        self.__learnCounter = LEARN
        self.iteration = 0

    def step(self):
        self.iteration += 1
        action = tf.argmax(self.estimator(self.state[np.newaxis, :])[0]).numpy()
        observation, reward, done, info = self.env.step(action)
        self.mem.add(self.state, action, reward, observation, done)
        self.state = observation

        if done:
            print(self.env.env.student._proficiency)  # hacks   , TODO: wandb support
            self.state = self.env.reset()

        if (self.iteration + 1) % self.batch_size == 0 and len(
            self.mem
        ) >= self.batch_size:
            print("learn")
            self.learn()
            self.__update_target()

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

        for i, (r, d, ns, a, s) in enumerate(
            zip(rewards, dones, next_states, actions, states)
        ):
            if d:
                real_q.append(tf.constant([r], dtype=tf.float32))
            else:
                next_state = tf.expand_dims(tf.constant(ns), axis=0)
                real_q.append(r + self.gamma * self.__get_target_q(next_state)[0])
            states_buff.append(s)
            actions_buff.append((i, a))
        self.estimator.train_step(
            tf.constant(states_buff), tf.constant(actions_buff), tf.stack(real_q)
        )

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
