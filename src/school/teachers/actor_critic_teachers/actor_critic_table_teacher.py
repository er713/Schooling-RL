from typing import List
import tensorflow_probability as tfp
import tensorflow as tf
from random import choice
import numpy as np

from . import _learn_main
from ... import Task
from ..models.actor_critic import *
from ..state_representation.table import TableTeacher
from ... import Result
from ..utils.dqn_structs import ExamResultsRecord


class ActorCriticTableTeacher(TableTeacher):
    def __init__(self,
                 nSkills: int,
                 tasks: List[Task],
                 cnn=False,
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
        super().__init__(nSkills, tasks, **kwargs)
        self.nTasks = len(tasks)
        self.mem = None  # tuple as <state, action, reward, next_state, done>
        if not cnn:
            self.actor = Actor(self.nTasks, verbose=self.verbose)
            self.critic = Critic()
        else:
            raise NotImplementedError("CNN don't support table representation xDD")
            # Sorry nie działa bo nie mamy nLasta xD
            # self.actor = ActorCNN(self.nTasks, verbose=verbose, nLast=nLast)
            # self.critic = CriticCNN(nTasks=self.nTasks, nLast=nLast)
        self.actor_opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.critic_opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def receive_result(self, result, last=False, reward=None) -> None:
        # Exam results need to be reduced in receive_exam_res
        super().receive_result(result, last, reward)
        # copy estimator weights to target estimator after noTargetIte iteration
        self.learn()

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
        self.mem = (state, action, reward, student_new_state, result.isExam)

    def get_action(self, state):
        logits = self.actor(state)
        action_probabilities = tfp.distributions.Categorical(logits=logits)
        action = action_probabilities.sample(sample_shape=())

        # if self.verbose:
        #     print(action)

        action = [task_ for task_ in self.tasks if task_.id == action][0]
        return action

    def learn(self) -> None:
        state, action, reward, next_state, done = self.mem
        if not next_state:
            next_state = tf.zeros((1, len(state)))
        else:
            next_state = tf.expand_dims(tf.convert_to_tensor(next_state), 0),
        _learn_main(self.actor, self.critic,
                    tf.expand_dims(tf.convert_to_tensor(state), 0),
                    tf.constant(action),
                    next_state,
                    tf.constant(reward, dtype=tf.float32),
                    tf.constant(done, dtype=tf.float32),
                    tf.constant(self.gamma, dtype=tf.float32),
                    self.actor_opt, self.critic_opt)
