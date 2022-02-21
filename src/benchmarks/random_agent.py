from collections import OrderedDict
from random import randint
from typing import Iterator, Tuple, List

import gym
import numpy as np
import torch
from pl_bolts.datamodules import ExperienceSourceDataset
from pl_bolts.models.rl.common.networks import MLP
from pytorch_lightning import LightningModule
from torch import Tensor, optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class RandomAgent(LightningModule):
    def __init__(self, env: str, batch_size: int):
        super().__init__()
        self.env = gym.make(env)
        self.env.reset()

        self.number_of_tasks = self.env.action_space.n
        self.batch_size = batch_size

        self.batch_proficiencies = []
        self.batch_exam_score_percentage = []
        self.batch_exam_score = []
        self.done_episodes = 0

        self.fake_module = MLP((128,), 1)

    def train_batch(self) -> Iterator[Tuple[np.ndarray, int, Tensor]]:
        self.batch_proficiencies = []
        self.batch_exam_score = []
        self.batch_exam_score_percentage = []
        for _ in range(self.batch_size):
            action = randint(0, self.number_of_tasks - 1)
            _, _, done, info = self.env.step(action)

            if done:
                self.done_episodes += 1
                self.batch_proficiencies.append(info["final_proficiencies"])
                self.batch_exam_score.append(info["exam_score"])
                self.batch_exam_score_percentage.append(info["exam_score_percentage"])
                self.env.reset()

        for idx in range(self.batch_size):
            yield [0]

    def training_step(self, batch: Tuple[Tensor], batch_idx: int) -> OrderedDict:
        proficiencies = np.array(self.batch_proficiencies)

        log = {
            "episodes": self.done_episodes,
            "avg_batch_exam_score": np.mean(self.batch_exam_score),
            "avg_batch_percentage_reward": np.mean(self.batch_exam_score_percentage),
            "avg_batch_mean_proficiency": proficiencies.mean(),
            "avg_batch_std_proficiency": proficiencies.std(axis=1).mean(),
        }
        self.log_dict(log, prog_bar=True, logger=True)
        return OrderedDict({"loss": torch.tensor(0.0, requires_grad=True)})

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = optim.Adam(self.fake_module.parameters(), lr=1e-3)
        return [optimizer]

    def _dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = ExperienceSourceDataset(self.train_batch)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size)
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self._dataloader()
