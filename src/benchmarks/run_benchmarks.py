import gym
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import ParameterGrid

from benchmarks.a2c import UpdatedA2C
from benchmarks.dqn import UpdatedDQN

param_grid = {
    "skills_quantity": [1, 2, 5, 10],
    "train_time_per_skill": [6, 8, 10, 12],
    "seed": [1, 2, 3, 4, 5],
    "env_name": ["gradesbook-v0"],
    # "env_name": ["gradesbook-v0", "gradeslist-v0"],
    # "model": ["a2c", "dqn"],
    "model": ["dqn"],
}
# iterations ale calculated to make the same number of episodes -> 10k
# episode len = training_time -> skills_quantity * train_time_per_skill

model_to_class = {"a2c": UpdatedA2C, "dqn": UpdatedDQN}
model_to_train_params = {
    "a2c": {
        "batch_size": 256,
        "epoch_len": 1,  # one batch per epoch, so epoch = iteration
    },
    "dqn": {
        "batch_size": 256,
        "replay_size": 256 * 5,
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
    wandb_logger = WandbLogger(project="schooling-rl", name="benchmarking")

    required_actions = episode_length * 20000
    trainer = Trainer(
        deterministic=True,
        logger=None,
        max_epochs=required_actions // model_params["batch_size"],
    )
    trainer.fit(model)
    break
