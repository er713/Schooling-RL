from argparse import ArgumentParser

import gym
from pl_bolts.models.rl import AdvantageActorCritic
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger


envs_data = {"skills_quantity": 1, "time_to_exam": 10}

gym.envs.register(
    id="gradesbook-v0",
    entry_point="environment:GradesBookEnvironment",
    kwargs=envs_data,
)
gym.envs.register(
    id="gradeslist-v0",
    entry_point="environment:GradesListEnvironment",
    kwargs=envs_data,
)

parser = ArgumentParser(add_help=False)
parser = Trainer.add_argparse_args(parser)
parser = AdvantageActorCritic.add_model_specific_args(parser)
args = parser.parse_args()

seed_everything(123)
model = AdvantageActorCritic(**args.__dict__)
wandb_logger = WandbLogger(project="schooling-rl", name=str(envs_data))
trainer = Trainer.from_argparse_args(args, deterministic=True, logger=wandb_logger)
trainer.fit(model)


# Can be run using:
# python3 script.py --env gradesbook-v0 --batch_size 256 --log_every_n 1
# python3 script.py --env gradeslist-v0 --batch_size 256 --log_every_n 1
