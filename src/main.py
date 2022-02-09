import fire
import gym

from actors.base_teacher import SimpleTeacher


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
    skills_quantity: int = 1,
    time_to_exam: int = 10,
    max_steps: int = 100,
):
    initialize_environments(skills_quantity, time_to_exam)
    actor = SimpleTeacher(env_name=env_name)
    for epoch in range(max_steps):
        actor.step()


if __name__ == "__main__":
    fire.Fire(train)
