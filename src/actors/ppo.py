from typing import Any, Tuple, List
import gym
from pl_bolts.utils import _GYM_AVAILABLE
from pl_bolts.models.rl.common.networks import MLP, ActorCategorical, ActorContinous

import numpy as np
from pl_bolts.models.rl.ppo_model import PPO
import torch
from torch import Tensor


class UpdatedPPO(PPO):
    def __init__(
        self,
        env: str,
        gamma: float = 0.99,
        lam: float = 0.95,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        max_episode_len: float = 200,
        batch_size: int = 512,
        steps_per_epoch: int = 2048,
        nb_optim_iters: int = 4,
        clip_ratio: float = 0.2,
        **kwargs: Any,
    ):
        super().__init__(env=env)

        if not _GYM_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError(
                "This Module requires gym environment which is not installed yet."
            )

        # Hyperparameters
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.steps_per_epoch = steps_per_epoch
        self.nb_optim_iters = nb_optim_iters
        self.batch_size = batch_size
        self.gamma = gamma
        self.lam = lam
        self.max_episode_len = max_episode_len
        self.clip_ratio = clip_ratio
        self.save_hyperparameters()

        self.env = gym.make(env)
        # value network
        self.critic = MLP(self.env.observation_space.shape, 1)
        # policy network (agent)
        if isinstance(self.env.action_space, gym.spaces.box.Box):
            act_dim = self.env.action_space.shape[0]
            actor_mlp = MLP(self.env.observation_space.shape, act_dim)
            self.actor = ActorContinous(actor_mlp, act_dim)
        elif isinstance(self.env.action_space, gym.spaces.discrete.Discrete):
            actor_mlp = MLP(self.env.observation_space.shape, self.env.action_space.n)
            self.actor = ActorCategorical(actor_mlp)
        else:
            raise NotImplementedError(
                "Env action space should be of type Box (continous) or Discrete (categorical). "
                f"Got type: {type(self.env.action_space)}"
            )

        self.batch_states = []
        self.batch_actions = []
        self.batch_adv = []
        self.batch_qvals = []
        self.batch_logp = []

        self.ep_rewards = []
        self.ep_values = []
        self.epoch_rewards = []

        self.episode_step = 0
        self.avg_ep_reward = 0
        self.avg_ep_len = 0
        self.avg_reward = 0

        self.done_episodes = 0
        self.batch_exam_score = []
        self.batch_exam_score_percentage = []
        self.batch_proficiency = []

        self.state = torch.FloatTensor(self.env.reset())

    def generate_trajectory_samples(
        self,
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """Contains the logic for generating trajectory data to train policy and value network.

        Yield:
           Tuple of Lists containing tensors for states, actions, log probs, qvals and advantage
        """

        self.batch_exam_score = []
        self.batch_exam_score_percentage = []
        self.batch_proficiency = []

        for step in range(self.steps_per_epoch):
            self.state = self.state.to(device=self.device)

            with torch.no_grad():
                pi, action, value = self(self.state)
                log_prob = self.actor.get_log_prob(pi, action)

            next_state, reward, done, info = self.env.step(action.cpu().numpy())

            self.episode_step += 1

            self.batch_states.append(self.state)
            self.batch_actions.append(action)
            self.batch_logp.append(log_prob)

            self.ep_rewards.append(float(reward))
            self.ep_values.append(value.item())

            self.state = torch.FloatTensor(next_state)

            epoch_end = step == (self.steps_per_epoch - 1)
            terminal = len(self.ep_rewards) == self.max_episode_len

            if epoch_end or done or terminal:
                # if trajectory ends abtruptly, boostrap value of next state
                if (terminal or epoch_end) and not done:
                    self.state = self.state.to(device=self.device)
                    with torch.no_grad():
                        _, _, value = self(self.state)
                        last_value = value.item()
                        steps_before_cutoff = self.episode_step
                else:
                    last_value = 0
                    steps_before_cutoff = 0
                    self.batch_exam_score.append(info["exam_score"])
                    self.batch_exam_score_percentage.append(
                        info["exam_score_percentage"]
                    )
                    self.batch_proficiency.append(info["final_proficiencies"])
                    self.done_episodes += 1

                # discounted cumulative reward
                self.batch_qvals += self.discount_rewards(
                    self.ep_rewards + [last_value], self.gamma
                )[:-1]
                # advantage
                self.batch_adv += self.calc_advantage(
                    self.ep_rewards, self.ep_values, last_value
                )
                # logs
                self.epoch_rewards.append(sum(self.ep_rewards))
                # reset params
                self.ep_rewards = []
                self.ep_values = []
                self.episode_step = 0
                self.state = torch.FloatTensor(self.env.reset())

            if epoch_end:
                train_data = zip(
                    self.batch_states,
                    self.batch_actions,
                    self.batch_logp,
                    self.batch_qvals,
                    self.batch_adv,
                )

                for state, action, logp_old, qval, adv in train_data:
                    yield state, action, logp_old, qval, adv

                self.batch_states.clear()
                self.batch_actions.clear()
                self.batch_adv.clear()
                self.batch_logp.clear()
                self.batch_qvals.clear()

                # logging
                self.avg_reward = sum(self.epoch_rewards) / self.steps_per_epoch

                # if epoch ended abruptly, exlude last cut-short episode to prevent stats skewness
                epoch_rewards = self.epoch_rewards
                if not done:
                    epoch_rewards = epoch_rewards[:-1]

                total_epoch_reward = sum(epoch_rewards)
                nb_episodes = len(epoch_rewards)

                self.avg_ep_reward = total_epoch_reward / nb_episodes
                self.avg_ep_len = (
                    self.steps_per_epoch - steps_before_cutoff
                ) / nb_episodes

                self.epoch_rewards.clear()

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx, optimizer_idx):
        """Carries out a single update to actor and critic network from a batch of replay buffer.

        Args:
            batch: batch of replay buffer/trajectory data
            batch_idx: not used
            optimizer_idx: idx that controls optimizing actor or critic network

        Returns:
            loss
        """
        state, action, old_logp, qval, adv = batch

        # normalize advantages
        adv = (adv - adv.mean()) / adv.std()

        # self.log("avg_ep_len", self.avg_ep_len, prog_bar=True, on_step=False, on_epoch=True)
        # self.log("avg_ep_reward", self.avg_ep_reward, prog_bar=True, on_step=False, on_epoch=True)
        # self.log("avg_reward", self.avg_reward, prog_bar=True, on_step=False, on_epoch=True)
        proficiencies = np.array(self.batch_proficiency)
        self.log_dict(
            {
                "episodes": self.done_episodes,
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

        if optimizer_idx == 0:
            loss_actor = self.actor_loss(state, action, old_logp, adv)
            # self.log("loss_actor", loss_actor, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            return loss_actor

        if optimizer_idx == 1:
            loss_critic = self.critic_loss(state, qval)
            # self.log("loss_critic", loss_critic, on_step=False, on_epoch=True, prog_bar=False, logger=True)

            return loss_critic

        raise NotImplementedError(
            f"Got optimizer_idx: {optimizer_idx}. Expected only 2 optimizers from configure_optimizers. "
            "Modify optimizer logic in training_step to account for this. "
        )
