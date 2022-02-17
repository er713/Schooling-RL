from collections import OrderedDict
from typing import Tuple

import gym
import numpy as np
import torch
from pl_bolts.datamodules.experience_source import Experience
from pl_bolts.losses.rl import dqn_loss
from pl_bolts.models.rl import DQN
from pl_bolts.models.rl.common.agents import ValueAgent
from pl_bolts.models.rl.common.networks import MLP
from torch import Tensor


class UpdatedDQN(DQN):
    def __init__(
        self,
        env: str,
        eps_start: float = 1.0,
        eps_end: float = 0.02,
        eps_last_frame: int = 150000,
        sync_rate: int = 1000,
        gamma: float = 0.99,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        replay_size: int = 100000,
        warm_start_size: int = 10000,
        avg_reward_len: int = 100,
        min_episode_reward: int = -21,
        seed: int = 123,
        batches_per_epoch: int = 1000,
        n_steps: int = 1,
        **kwargs,
    ):
        """
        Args:
            env: gym environment tag
            eps_start: starting value of epsilon for the epsilon-greedy exploration
            eps_end: final value of epsilon for the epsilon-greedy exploration
            eps_last_frame: the final frame in for the decrease of epsilon. At this frame espilon = eps_end
            sync_rate: the number of iterations between syncing up the target network with the train network
            gamma: discount factor
            learning_rate: learning rate
            batch_size: size of minibatch pulled from the DataLoader
            replay_size: total capacity of the replay buffer
            warm_start_size: how many random steps through the environment to be carried out at the start of
                training to fill the buffer with a starting point
            avg_reward_len: how many episodes to take into account when calculating the avg reward
            min_episode_reward: the minimum score that can be achieved in an episode. Used for filling the avg buffer
                before training begins
            seed: seed value for all RNG used
            batches_per_epoch: number of batches per epoch
            n_steps: size of n step look ahead
        """
        super(DQN, self).__init__()  # dirty hack to omit calling DQN init
        # Environment
        self.exp = None
        self.env = gym.make(env)
        # self.test_env = gym.make(env)

        self.obs_shape = self.env.observation_space.shape
        self.n_actions = self.env.action_space.n

        # Model Attributes
        self.buffer = None
        self.dataset = None

        self.net = None
        self.target_net = None
        self.build_networks()

        self.agent = ValueAgent(
            self.net,
            self.n_actions,
            eps_start=eps_start,
            eps_end=eps_end,
            eps_frames=eps_last_frame,
        )

        # Hyperparameters
        self.sync_rate = sync_rate
        self.gamma = gamma
        self.lr = learning_rate
        self.batch_size = batch_size
        self.replay_size = replay_size
        self.warm_start_size = warm_start_size
        self.batches_per_epoch = batches_per_epoch
        self.n_steps = n_steps

        self.save_hyperparameters()

        # Metrics
        self.total_episode_steps = [0]
        self.total_rewards = [0]
        self.done_episodes = 0
        self.total_steps = 0
        self.batch_states = []
        self.batch_actions = []
        self.batch_rewards = []
        self.batch_exam_score = []
        self.batch_masks = []
        self.batch_proficiencies = []
        self.batch_exam_score_percentage = []

        # Average Rewards
        self.avg_reward_len = avg_reward_len

        for _ in range(avg_reward_len):
            self.total_rewards.append(
                torch.tensor(min_episode_reward, device=self.device)
            )

        self.avg_rewards = float(np.mean(self.total_rewards[-self.avg_reward_len :]))

        self.state = self.env.reset()

    def build_networks(self) -> None:
        """Initializes the DQN train and target networks."""
        self.net = MLP(self.obs_shape, self.n_actions)
        self.target_net = MLP(self.obs_shape, self.n_actions)

    def train_batch(
        self,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Contains the logic for generating a new batch of data to be passed to the DataLoader.

        Returns:
            yields a Experience tuple containing the state, action, reward, done and next_state.
        """
        episode_reward = 0
        episode_steps = 0

        self.batch_states = []
        self.batch_actions = []
        self.batch_rewards = []
        self.batch_exam_score = []
        self.batch_masks = []
        self.batch_proficiencies = []
        self.batch_exam_score_percentage = []

        for i in range(self.batch_size):
            self.total_steps += 1
            action = self.agent(self.state, self.device)

            next_state, r, is_done, info = self.env.step(action[0])

            self.batch_states.append(self.state)
            self.batch_actions.append(action[0])
            self.batch_rewards.append(r)
            self.batch_masks.append(is_done)

            episode_reward += r
            episode_steps += 1

            exp = Experience(
                state=self.state,
                action=action[0],
                reward=r,
                done=is_done,
                new_state=next_state,
            )
            self.agent.update_epsilon(self.global_step * self.batch_size + i)
            self.buffer.append(exp)
            self.state = next_state

            if is_done:
                self.done_episodes += 1
                self.batch_exam_score.append(info["exam_score"])
                self.batch_exam_score_percentage.append(info["exam_score_percentage"])
                self.batch_proficiencies.append(info["final_proficiencies"])
                self.total_rewards.append(episode_reward)
                self.total_episode_steps.append(episode_steps)
                self.avg_rewards = float(
                    np.mean(self.total_rewards[-self.avg_reward_len :])
                )
                self.state = self.env.reset()
                episode_steps = 0
                episode_reward = 0

        states, actions, rewards, dones, new_states = self.buffer.sample(
            self.batch_size
        )
        for idx, _ in enumerate(dones):
            yield states[idx], actions[idx], rewards[idx], dones[idx], new_states[idx]

    def on_epoch_end(self) -> None:
        proficiencies = np.array(self.batch_proficiencies)
        self.log_dict(
            {
                "episodes": self.done_episodes,
                "epsilon": self.agent.epsilon,
                "avg_batch_exam_score": np.mean(self.batch_exam_score),
                "avg_batch_percentage_reward": np.mean(
                    self.batch_exam_score_percentage
                ),
                "avg_batch_mean_proficiency": np.mean(proficiencies),
                "avg_batch_std_proficiency": proficiencies.std(axis=1).mean(),
            },
            logger=True,
            prog_bar=True,
        )

    def training_step(self, batch: Tuple[Tensor, Tensor], _) -> OrderedDict:
        """Carries out a single step through the environment to update the replay buffer. Then calculates loss
        based on the minibatch recieved.

        Args:
            batch: current mini batch of replay data
            _: batch number, not used

        Returns:
            Training loss and log metrics
        """

        # calculates training loss
        loss = dqn_loss(batch, self.net, self.target_net, self.gamma)

        if self._use_dp_or_ddp2(self.trainer):
            loss = loss.unsqueeze(0)

        # Soft update of target network
        if self.global_step % self.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        # if self.done_episodes % (self.batch_size * self.batches_per_epoch) == 0:
        #     proficiencies = np.array(self.batch_proficiencies)
        #     self.log_dict(
        #         {
        #             "episodes": self.done_episodes,
        #             "avg_batch_exam_score": np.mean(self.batch_exam_score),
        #             "avg_batch_percentage_reward": np.mean(self.batch_exam_score_percentage),
        #             "avg_batch_mean_proficiency": np.mean(proficiencies),
        #             "avg_batch_std_proficiency": proficiencies.std(axis=1).mean(),
        #         },
        #         logger=True,
        #         prog_bar=True,
        #     )
        # OLD
        # "total_reward": self.total_rewards[-1],
        # "avg_reward": self.avg_rewards,
        # "train_loss": loss,
        # "episode_steps": self.total_episode_steps[-1],

        return OrderedDict(
            {
                "loss": loss,
            }
        )
