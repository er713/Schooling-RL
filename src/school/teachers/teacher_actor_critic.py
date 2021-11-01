from typing import List
import tensorflow_probability as tfp
import tensorflow as tf
import losses
from . import Teacher
from .. import Task
from .models.actor_critic import *



class TeacherActorCritic(Teacher):
    def __init__(self, nSkills: int, tasks: List[Task], gamma: float, nLast: int, learning_rate: int, **kwargs):
        super().__init__(nSkills, tasks, **kwargs)
        self.actor = Actor(len(tasks))
        self.critic = Critic()
        self.actor_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.critic_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.gamma = gamma
        self.nLast = nLast
        self.results = dict()

    def _get_state(self, idStudent: int, shift: int = 0) -> List[int]:
        """
        Function for getting state out of history (self.results) for specified student.
        :param idStudent: Student ID
        :param shift: Shift to the past/how many recent Results skip. Has to be positive.
        :return: State - list of int/float of shape 3*self.nLast
        """
        student_results = self.results.get(idStudent, [])[-(self.nLast + shift):(len(self.tasks) - shift)]
        state = []
        for result in student_results:
            state.append(result.task.id)
            state.append(list(result.task.taskDifficulties.keys())[0])
            state.append(result.mark)
        while len(state) < self.nLast * 3:
            [state.append(-100) for _ in
             range(3)]  # TODO: ustalić co będzie pustym elementem/do wypełnienia brakujących wartości
        return state

    def choose_task(self, student) -> Task:
        state = self._get_state(student)
        logits = self.actor(state)
        action_probabilities = tfp.distributions.Categorical(logits=logits)
        action = action_probabilities.sample()
        task = [task for task in self.tasks if task.id == action][0]
        return task

    def receive_result(self, result, reward=None, last=False) -> None:
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
            self._learn(self._get_state(student, shift=1), result.task.id, self._get_state(student), reward, done)
            if reward > 0:
                self.results[student] = []  # remove student history after exam

    """
    Mając state wykonaj akcje a, zaobserwuj nagrodę reward i następnik next_state
    """

    def _learn(self, state: List[int], action: int, next_state: List[int], reward: int, done: int) -> None:
        """
        Dokumentacja
        """
        with tf.GradientTape() as tape:
            q = self.critic(state)
            q_next = self.critic(next_state)
            logits = self.actor(state)

            δ = reward + self.gamma * q_next * (1 - done) - q


            actor_loss = losses.actor_loss(logits,action,δ)
            critic_loss = δ ** 2  # MSE?

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)

        self.actor_opt.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        self.critic_opt.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
