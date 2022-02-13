import gym
import numpy as np
import tensorflow as tf
from gym import Env
from tensorflow.keras.layers import Dense

from utils.constants import MEM_SIZE, BATCH_SIZE, GAMMA, TARGET_ITER, LEARN
from utils.replay_buffer import ReplayBuffer


class MLP(tf.keras.Model):
    def __init__(self, inputSize, outputSize):
        super().__init__(name="dqn")
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.loss = tf.keras.losses.Huber()

        self.model = tf.keras.Sequential(
            [
                Dense(inputSize, activation="relu", input_shape=(inputSize,)),
                Dense(128, activation="relu"),
                Dense(outputSize),
            ]
        )

    @tf.function
    def call(self, inputs, training):
        print("Retracing calla")
        """ Given a state, return Q values of actions"""
        return self.model(inputs, training=training)

    def copy_weights(self, otherModel):
        model = otherModel.model
        for idx, layer in enumerate(model.layers):
            weights = layer.get_weights()
            self.model.layers[idx].set_weights(weights)

    @tf.function
    def train_step(self, states, actions, realQs):
        print("Retracing train_stepa")
        with tf.GradientTape() as tape:
            qPred = self(states, training=True)
            qPred = tf.gather_nd(qPred, actions)
            lossValue = self.loss(realQs, qPred)
        variables = self.model.trainable_variables
        grads = tape.gradient(lossValue, variables)
        self.optimizer.apply_gradients(zip(grads, variables))


class DQN:
    def __init__(
        self,
        env_name: str,
        mem_size=MEM_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        learn_every_iters=LEARN,
    ):
        self.env: Env = gym.make(env_name)
        self.state = self.env.reset()

        self.mem_size = mem_size
        self.batch_size = batch_size
        self.gamma = gamma

        self.modelInputSize = self.env.observation_space.shape[0]
        self.modelOutputSize = self.env.action_space.n
        self.estimator = MLP(self.modelInputSize, self.modelOutputSize)
        self.targetEstimator = MLP(self.modelInputSize, self.modelOutputSize)

        self.mem = ReplayBuffer(1, self.mem_size, self.batch_size)

        self.learn_every_iters = learn_every_iters
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
            self.learn()
            self.__update_target()

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
        self.targetEstimator.copy_weights(self.estimator)

    @tf.function
    def __get_target_q(self, state: tf.Tensor):
        modelOutput = self.estimator(state)
        bestActionIdx = tf.expand_dims(tf.argmax(modelOutput, axis=1), axis=1)
        targetQVector = self.targetEstimator(state)
        targetQ = tf.gather(targetQVector, bestActionIdx, axis=1, batch_dims=1)
        return targetQ
