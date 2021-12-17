import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf
from typing import List

from .. import TeacherAllHistoryRNN
from .. import losses
from ..models import RNNWrapper
from ... import Task
from ..models.actor_critic import *


class ActorCriticAllHistoryRNNTeacher(TeacherAllHistoryRNN):
    def __init__(self,
                 nSkills: int,
                 tasks: List[Task],
                 nStudents: int,
                 cnn: bool = False,
                 verbose: bool = False,
                 task_embedding_size: int = 10,
                 rnn_units: int = None,
                 *args, **kwargs):
        if rnn_units is None:
            rnn_units = len(tasks) // 7
        super().__init__(nSkills, tasks, task_embedding_size, rnn_units, **kwargs)

        self._actor = Actor(self.nTasks)
        self._critic = Critic()
        self.ac = RNNWrapper(self.rnn_units, self.nTasks, task_embedding_size, [self._actor, self._critic])

        self.ac_opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        # self.critic_opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.nStudents = nStudents
        self.verbose = verbose
        # if verbose:
        self.choices = np.zeros((self.nTasks,), dtype=np.int_)

    def get_action(self, state):
        logits, state = self.ac.get_specific_call(state[0], state[1])
        self.last_rnn_state = state
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
            if self.iteration_st % (self.nStudents*10) == 0:  # TODO: zmienić na bardziej sensowne
                print(self.iteration_st)  # TODO: potem zmazać
                print('epsilon:', self.epsilon)
                print('variety:\n',
                      np.reshape(self.choices, (self.choices.shape[0] // 7, 7)))
                self.choices = np.zeros((len(self.tasks),), dtype=np.int_)

        _learn_main(self.ac, state, tf.constant(action), next_state,
                    tf.constant(1000 * reward + 0.001, dtype=tf.float32),
                    tf.constant(done, dtype=tf.float32), tf.constant(self.gamma, dtype=tf.float32), self.ac_opt)


@tf.function
def _learn_main(ac: tf.keras.Model, state: tf.Tensor, action: tf.Tensor,
                next_state: tf.Tensor,
                reward: tf.Tensor, done: tf.Tensor, gamma: tf.Tensor, ac_opt) -> None:
    """
    Dokumentacja
    """
    with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
        q, st = ac.get_specific_call(state[0], state[1], 1)
        q_next, _ = ac.get_specific_call(next_state[0], st, 1)
        logits, _ = ac.get_specific_call(state[0], state[1], 0)

        δ = reward + gamma * q_next * (1 - done) - q  # this works w/o tf.function
        # δ = float(reward) + float(gamma * q_next * (1 - done)) - float(q)  # float only for tf.function

        actor_loss = losses.actor_loss(logits, action, δ)
        critic_loss = δ ** 2  # MSE?

    critic_grads = critic_tape.gradient(critic_loss, ac.trainable_variables)
    ac_opt.apply_gradients(zip(critic_grads, ac.trainable_variables))

    actor_grads = actor_tape.gradient(actor_loss, ac.trainable_variables)
    ac_opt.apply_gradients(zip(actor_grads, ac.trainable_variables))

