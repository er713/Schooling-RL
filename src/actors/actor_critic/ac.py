from __future__ import annotations
import gym
import numpy as np
from gym import Env
import tensorflow as tf
from keras.optimizers import adam_v2
from tensorflow_probability.python.distributions import Categorical
import wandb

from actors.actor_critic.network import ACNetwork


def _cut_state_for_gradeslist(x, n, timestamp, tasks):
    sk_p1 = tasks + 1
    t0 = max(0, timestamp - n + 1)
    cut = x[t0 * sk_p1:(timestamp + 1) * sk_p1]
    return np.reshape(np.concatenate([np.zeros(n * sk_p1 - len(cut)), cut], 0), (1, n * sk_p1))


class AcorCriticTeacher:
    """

    """
    _state_cutting_func = {
        "gradesbook-v0": lambda x, *args: [x],
        "gradeslist-v0": _cut_state_for_gradeslist
    }

    def __init__(
            self, env_name, project, entity, gamma, start_epsilon, min_epsilon, max_epsilon,
            epsilon_decay, n_last, learning_rate=5.e-3
    ):
        # Initialise env
        self.env: Env = gym.make(env_name)
        state = self.env.reset()

        # Initialise actors and optimizers
        self.actor_nn = ACNetwork(output_size=self.env.number_of_tasks)
        self.critic_nn = ACNetwork(output_size=1)
        self.actor_opt = adam_v2.Adam(learning_rate=learning_rate)
        self.critic_opt = adam_v2.Adam(learning_rate=learning_rate)

        # Other
        self.gamma = tf.constant(gamma)
        self.cut_state = self._state_cutting_func[env_name]
        self.n_last = n_last
        self.epsilon = start_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.actions = []
        self.state = tf.constant(self.cut_state(state, self.n_last,
                                                self.env.iteration, self.env.number_of_tasks))

        # Logger
        self.wandb = wandb.init(project=project, entity=entity,
                                config={
                                    "learning_rate": learning_rate,
                                    "gamma": float(self.gamma),
                                    "env": env_name,
                                    "start_epsilon": self.epsilon,
                                    "min_epsilon": self.min_epsilon,
                                    "max_epsilon": self.max_epsilon,
                                    "epsilon_decay": self.epsilon_decay,
                                    "n_last": self.n_last
                                })

    def step(self):
        state = self.state
        # Predict next action
        if np.random.random() > self._get_epsilon():  # from actor
            logits = self.actor_nn(self.state)
            action = Categorical(logits=logits).sample()
        else:  # random
            action = tf.constant(self.env.action_space.sample())

        # Feed environment
        self.actions.append(int(action))
        observation, reward, done, info = self.env.step(self.actions[-1])

        # Learn AC
        observation = tf.constant(self.cut_state(observation, self.n_last,
                                                 self.env.iteration, self.env.number_of_tasks))
        reward = tf.constant(0. if not done else float(reward))
        done = tf.constant(1. if done else 0.)
        _learn_main(self.actor_nn, self.critic_nn, state, action, observation, reward,
                    done, self.gamma, self.actor_opt, self.critic_opt)
        self.state = observation

        # Reset if end
        if done:
            p = {f"proficiency_{i}": p for i, p in enumerate(self.env.env.student._proficiency)}
            p.update({f"actions_{i}": a for i, a in enumerate(self.actions)})
            p.update({"epsilon": self._get_epsilon(), "exam_score": info['exam_score_percentage']})
            wandb.log(p)
            print(self.env.env.student._proficiency)  # hacks
            self.env.reset()
            self.actions = []
            self.epsilon *= self.epsilon_decay

    def _get_epsilon(self):
        return max(self.min_epsilon, min(self.epsilon, self.max_epsilon))


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

        δ = reward + gamma * tf.cast(q_next, tf.float32) * (1. - done) - tf.cast(q, tf.float32)

        actor_loss = loss(logits, action, δ)
        critic_loss = δ ** 2  # MSE?

    actor_grads = actor_tape.gradient(actor_loss, actor.trainable_variables)
    critic_grads = critic_tape.gradient(critic_loss, critic.trainable_variables)

    actor_opt.apply_gradients(zip(actor_grads, actor.trainable_variables))
    critic_opt.apply_gradients(zip(critic_grads, critic.trainable_variables))


@tf.function
def loss(logits, action, d):
    cat = Categorical(logits=logits)
    log_prob = cat.log_prob(action)
    return -log_prob * d
