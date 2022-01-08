import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf
from typing import List

from .. import losses
from .. import TeacherAllHistory
from ..layers import AllHistoryCNN
from ... import Task
from ..models.actor_critic import *


class ActorCriticAllHistoryCNNTeacher(TeacherAllHistory):
    def __init__(self,
                 nSkills: int,
                 tasks: List[Task],
                 cnn: bool = False,
                 task_embedding_size: int = 10,
                 base_history: int = 7,
                 filters: int = 5,
                 *args, **kwargs):
        super().__init__(nSkills, tasks, task_embedding_size, base_history, **kwargs)

        self._actor = Actor(self.nTasks)
        self._critic = Critic()
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

        _learn_main(self.actor, self.critic, state, tf.constant(action), next_state,
                    tf.constant(1000 * reward + 0.001, dtype=tf.float32),
                    tf.constant(done, dtype=tf.float32), tf.constant(self.gamma, dtype=tf.float32), self.actor_opt,
                    self.critic_opt)


@tf.function
def _learn_main(actor: tf.keras.Model, critic: tf.keras.Model, state: tf.Tensor, action: tf.Tensor,
                next_state: tf.Tensor,
                reward: tf.Tensor, done: tf.Tensor, gamma: tf.Tensor, actor_opt, critic_opt) -> None:
    """
    Dokumentacja
    """
    with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
        q = critic(state)
        q_next = critic(next_state)
        logits = actor(state)

        δ = reward + gamma * q_next * (1 - done) - q  # this works w/o tf.function
        # δ = float(reward) + float(gamma * q_next * (1 - done)) - float(q)  # float only for tf.function

        actor_loss = losses.actor_loss(logits, action, δ)
        critic_loss = δ ** 2  # MSE?

    actor_grads = actor_tape.gradient(actor_loss, actor.trainable_variables)
    critic_grads = critic_tape.gradient(critic_loss, critic.trainable_variables)

    actor_opt.apply_gradients(zip(actor_grads, actor.trainable_variables))
    critic_opt.apply_gradients(zip(critic_grads, critic.trainable_variables))
