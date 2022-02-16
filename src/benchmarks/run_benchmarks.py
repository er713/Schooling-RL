from argparse import ArgumentParser

import gym
from pl_bolts.models.rl import AdvantageActorCritic, DQN
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger

from sklearn.model_selection import ParameterGrid

param_grid = {
    "skills_quantity": [1, 2, 5, 10],
    "train_time_per_skill": [4, 8, 12, 16],
    "seed": [1, 2, 3, 4, 5],
    "env_name": ["gradesbook-v0"],
    # "env_name": ["gradesbook-v0", "gradeslist-v0"],
    "model": ["ac", "dqn"],
}
model_to_class = {"a2c": AdvantageActorCritic, "dqn": DQN}
model_to_train_params = {"a2c": {}, "dqn": {}}
for params in ParameterGrid(param_grid):
    gym.envs.register(
        id="gradesbook-v0",
        entry_point="environment:GradesBookEnvironment",
        kwargs={
            "skills_quantity": params["skills_quantity"],
            "time_to_exam": params["skills_quantity"] * params["train_time_per_skill"],
        },
    )
    gym.envs.register(
        id="gradeslist-v0",
        entry_point="environment:GradesListEnvironment",
        kwargs={
            "skills_quantity": params["skills_quantity"],
            "time_to_exam": params["skills_quantity"] * params["train_time_per_skill"],
        },
    )

    seed_everything(params["seed"])
    model_class = model_to_class[params["model"]]
    model_params = model_to_train_params[params["model"]]

    model = model_class(env=params["env_name"], **model_params)
    wandb_logger = WandbLogger(project="schooling-rl", name="benchmarking")
    trainer = Trainer(deterministic=True, logger=None)
    trainer.fit(model)
    break
