import gym
import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import ParameterGrid

from actors.a2c import UpdatedA2C
from actors.ppo import UpdatedPPO
from actors.random_agent import RandomAgent

param_grid = {
    "skills_quantity": [1, 3, 5, 7],
    "model": ["random", "ppo", "a2c"],
    "train_time_per_skill": [10],
    "seed": [0, 1, 2, 3, 4],
    "env_name": ["gradesbook-v0"],
}
# iterations ale calculated to make the same number of episodes -> 10k
# episode len = training_time -> skills_quantity * train_time_per_skill

model_to_class = {"a2c": UpdatedA2C, "random": RandomAgent, "ppo": UpdatedPPO}
model_to_train_params = {
    "a2c": {
        "batch_size": 256,
        "epoch_len": 1,  # one batch per epoch, so epoch = iteration
    },
    "ppo": {
        "batch_size": 512,
        "steps_per_epoch": 512,
        "nb_optim_iters": 1,
    },
    "random": {"batch_size": 256},
}

for params in ParameterGrid(param_grid):
    episode_length = params["skills_quantity"] * params["train_time_per_skill"]

    gym.envs.register(
        id="gradesbook-v0",
        entry_point="environment.environment:GradesBookEnvironment",
        kwargs={
            "skills_quantity": params["skills_quantity"],
            "time_to_exam": episode_length,
        },
    )

    seed_everything(params["seed"])
    model_class = model_to_class[params["model"]]
    model_params = model_to_train_params[params["model"]]

    model = model_class(env=params["env_name"], **model_params)
    wandb_logger = WandbLogger(project="schooling-rl", name="benchamrks(6)")
    wandb_logger.log_hyperparams(params)

    required_actions = episode_length * 25000
    trainer = Trainer(
        deterministic=False,
        logger=wandb_logger,
        max_epochs=required_actions // model_params["batch_size"],
        gpus=0,
    )
    trainer.fit(model)
    wandb.finish()
