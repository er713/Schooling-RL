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
        gamma: float = 0.99,
        learning_rate: float = 5.e-3,
        start_epsilon: float = .9,
        min_epsilon: float = .03,
        max_epsilon: float = .9,
        epsilon_decay: float = .99965,
        n_last: int = 10,
):
    initialize_environments(skills_quantity, time_to_exam)
    actor = AcorCriticTeacher(env_name=env_name, project="SchoolingRL", entity="eryk", gamma=gamma,
                              learning_rate=learning_rate, start_epsilon=start_epsilon, min_epsilon=min_epsilon,
                              max_epsilon=max_epsilon, epsilon_decay=epsilon_decay, n_last=n_last)
    actor.wandb.config['skills_quantity'] = skills_quantity
    actor.wandb.config['time_to_exam'] = time_to_exam
    actor.wandb.config['max_steps'] = max_steps

    try:
        for epoch in range(max_steps):
            actor.step()
    except KeyboardInterrupt:
        pass
    finally:
        actor.wandb.finish()


if __name__ == "__main__":
    fire.Fire(train)
