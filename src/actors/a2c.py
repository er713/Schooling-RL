from collections import OrderedDict
from typing import Iterator, Tuple

import numpy as np
from pl_bolts.models.rl import AdvantageActorCritic
from torch import Tensor


class UpdatedA2C(AdvantageActorCritic):
    def __init__(self, env: str, **kwargs):
        super().__init__(env, **kwargs)
        self.batch_proficiencies = []
        self.batch_exam_score_percentage = []
        self.batch_exam_score = []

    def train_batch(self) -> Iterator[Tuple[np.ndarray, int, Tensor]]:
        """Contains the logic for generating a new batch of data to be passed to the DataLoader.

        Returns:
            yields a tuple of Lists containing tensors for
            states, actions, and returns of the batch.

        Note:
            This is what's taken by the dataloader:
            states: a list of numpy array
            actions: a list of list of int
            returns: a torch tensor
        """
        self.batch_states = []
        self.batch_actions = []
        self.batch_rewards = []
        self.batch_exam_score = []
        self.batch_masks = []
        self.batch_proficiencies = []
        self.batch_exam_score_percentage = []
        for _ in range(self.hparams.batch_size):
            action = self.agent(self.state, self.device)[0]

            next_state, reward, done, info = self.env.step(action)
            self.batch_rewards.append(reward)
            self.batch_actions.append(action)
            self.batch_states.append(self.state)
            self.batch_masks.append(done)
            self.state = next_state
            self.episode_reward += reward

            if done:
                self.done_episodes += 1
                self.batch_proficiencies.append(info["final_proficiencies"])
                self.batch_exam_score.append(info["exam_score"])
                self.batch_exam_score_percentage.append(info["exam_score_percentage"])
                self.state = self.env.reset()
                self.total_rewards.append(self.episode_reward)
                self.episode_reward = 0
                self.avg_rewards = float(
                    np.mean(self.total_rewards[-self.avg_reward_len :])
                )

        _, last_value = self.forward(self.state)

        returns = self.compute_returns(self.batch_rewards, self.batch_masks, last_value)
        for idx in range(self.hparams.batch_size):
            yield self.batch_states[idx], self.batch_actions[idx], returns[idx]

    def training_step(
        self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int
    ) -> OrderedDict:
        """Perform one actor-critic update using a batch of data.

        Args:
            batch: a batch of (states, actions, returns)
        """
        states, actions, returns = batch
        loss = self.loss(states, actions, returns)
        proficiencies = np.array(self.batch_proficiencies)

        log = {
            "episodes": self.done_episodes,
            "avg_batch_exam_score": np.mean(self.batch_exam_score),
            "avg_batch_percentage_reward": np.mean(self.batch_exam_score_percentage),
            "avg_batch_mean_proficiency": proficiencies.mean(),
            "avg_batch_std_proficiency": proficiencies.std(axis=1).mean(),
        }
        self.log_dict(log, prog_bar=True, logger=True)
        return OrderedDict({"loss": loss})
