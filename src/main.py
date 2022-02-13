import fire
import gym

from actors.dqn import DQN
from actors.simple_teacher import SimpleTeacher

actor_factory = {"simple-teacher": SimpleTeacher, "dqn": DQN}


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
    actor_name: str = "simple-teacher",
    skills_quantity: int = 3,
    time_to_exam: int = 25,
    max_steps: int = 2000,
):
    """
    :param env_name: One of [gradesbook-v0, gradeslist-v0]
    :param actor_name: One of [simple-teacher, dqn]
    """
    initialize_environments(skills_quantity, time_to_exam)

    actor_class = actor_factory[actor_name]
    actor = actor_class(env_name=env_name)

    for epoch in range(max_steps):
        actor.step()


if __name__ == "__main__":
    fire.Fire(train)
