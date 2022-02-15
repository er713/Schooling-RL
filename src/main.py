import fire
import gym
import wandb

from actors.base_teacher import SimpleTeacher
from actors.actor_critic.ac import AcorCriticTeacher


def initialize_environments(skills_quantity: int = 1, time_to_exam: int = 10):
    envs_data = {"skills_quantity": skills_quantity, "time_to_exam": time_to_exam}

    gym.envs.register(
        id="gradesbook-v0",
        entry_point="environment.environment:GradesBookEnvironment",
        kwargs=envs_data,
    )
    gym.envs.register(
        id="gradeslist-v0",
        entry_point="environment.environment:GradesListEnvironment",
        kwargs=envs_data,
    )


def train(
    env_name: str = "gradesbook-v0",
    skills_quantity: int = 3,
    time_to_exam: int = 25,
    max_steps: int = 2000,
):
    initialize_environments(skills_quantity, time_to_exam)
    actor = AcorCriticTeacher(env_name=env_name)
    wr = wandb.init(project="SchoolingRL", entity="er713")

    for epoch in range(max_steps):
        actor.step()
    wr.finish()


if __name__ == "__main__":
    fire.Fire(train)
