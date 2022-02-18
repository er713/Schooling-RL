import gym
import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import ParameterGrid

from benchmarks.a2c import UpdatedA2C
from benchmarks.dqn import UpdatedDQN
from benchmarks.ppo import UpdatedPPO

param_grid = {
    "skills_quantity": [1, 2, 5, 10],
    "train_time_per_skill": [6, 8, 10, 12],
    "seed": [1, 2, 3, 4, 5],
    # "env_name": ["gradesbook-v0"],
    "env_name": ["gradesbook-v0", "gradeslist-v0"],
    # "model": ["a2c", "dqn"],
    "model": ["ppo"],
}
# iterations ale calculated to make the same number of episodes -> 10k
# episode len = training_time -> skills_quantity * train_time_per_skill

model_to_class = {"a2c": UpdatedA2C, "dqn": UpdatedDQN, "ppo": UpdatedPPO}
model_to_train_params = {
    "a2c": {
        "batch_size": 256,
        "epoch_len": 1,  # one batch per epoch, so epoch = iteration
    },
    "dqn": {
        "batch_size": 256,
        "replay_size": 256 * 5,
        "min_episode_reward": 0,
        "eps_last_frame": 80_000,
        "sync_rate": 16,
        "n_step": 5,
    },
    "ppo": {
        "batch_size": 256,
        "max_episode_len": 10,
        "steps_per_epoch": 1024,
    },
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
    gym.envs.register(
        id="gradeslist-v0",
        entry_point="environment.environment:GradesListEnvironment",
        kwargs={
            "skills_quantity": params["skills_quantity"],
            "time_to_exam": episode_length,
        },
    )

    seed_everything(params["seed"])
    model_class = model_to_class[params["model"]]
    model_params = model_to_train_params[params["model"]]

    model = model_class(env=params["env_name"], **model_params)
    wandb_logger = WandbLogger(project="schooling-rl", name="benchmarking_ppo")
    wandb_logger.log_hyperparams(params)

    required_actions = episode_length * 20000
    if params["model"] == "dqn":
        model_params["eps_last_frame"] = int(0.8 * required_actions)
    elif params["model"] == "ppo":
        model_params["max_episode_len"] = episode_length

    trainer = Trainer(
        deterministic=False,
        logger=None,
        max_epochs=required_actions // model_params["batch_size"],
        gpus=1,
    )
    trainer.fit(model)
    wandb.finish()
    # break
