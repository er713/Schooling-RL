import gym
import numpy as np
from gym import Env


class SimpleTeacher:
    """
    The teacher estimates the difficulty of the task based on numbers of correctly and not correctly solved tasks.
    The teacher always gives the student the task with the highest estimate for the least trained skills.
    """

    def __init__(self, env_name):
        self.env: Env = gym.make(env_name)
        self.env.reset()

        self.skills_quantity = self.env.skills_quantity
        self.number_of_difficulties = self.env.number_of_difficulties
        self.number_of_tasks = self.env.action_space.n

        self.task_successes = np.zeros(self.number_of_tasks)
        self.task_attempts = np.zeros(self.number_of_tasks) + 1
        self.skills_training_progress = np.zeros(self.skills_quantity)

        update_strategies = {
            "gradesbook-v0": self._update_state_using_grades_book,
            "gradeslist-v0": self._update_state_using_grades_list,
        }
        self.update_strategy = update_strategies[env_name]

    def step(self):
        least_trained_skill = np.argmin(self.skills_training_progress)
        self.skills_training_progress[least_trained_skill] += 1

        task_offset = least_trained_skill * self.number_of_difficulties
        tasks_range = slice(task_offset, task_offset + self.number_of_difficulties)

        estimated_difficulties = self.task_successes[tasks_range] / (
            self.task_attempts[tasks_range] + self.task_successes[tasks_range]
        )
        action = np.argmin(estimated_difficulties) + task_offset
        observation, _, done, _ = self.env.step(action)

        if done:
            print(self.env.env.student._proficiency)  # hacks   , TODO: wandb support

        self.update_strategy(observation, done)

        if done:
            self.env.reset()
            self.skills_training_progress = np.zeros(self.skills_quantity)

    def _update_state_using_grades_book(
        self, observation: np.array, done: bool
    ) -> None:
        """Update only during exam, just for simplification"""
        if done:
            observation = observation.reshape(2, -1)
            self.task_successes += observation[1]
            self.task_attempts += observation.sum(axis=0)

    def _update_state_using_grades_list(
        self, observation: np.array, done: bool
    ) -> None:
        """Update only during exam, just for simplification"""
        if done:
            all_observations = observation.reshape(-1, self.number_of_tasks + 1)
            is_solved_vector = all_observations[:, 0]
            one_hot_tasks = all_observations[:, 1:]

            self.task_successes += one_hot_tasks[is_solved_vector.astype(bool)].sum(
                axis=0
            )
            self.task_attempts += one_hot_tasks.sum(axis=0)
