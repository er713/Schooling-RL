from __future__ import annotations
from typing import Dict, Tuple
import numpy as np


class Task:
    def __init__(self, task_difficulties: Dict[int, float]) -> None:
        """
        :param task_difficulties: The dictionary of difficulty for skill
        """
        self.task_difficulties: Dict[int, float] = task_difficulties

    @classmethod
    def generate_random_task(
        cls,
        n_skills: int,
        min_skills_assigned: int = 1,
        max_skills_assigned: int = None,
        difficulties_range: Tuple[float, float] = (-3, 3),
    ) -> Task:
        """
        Create random Task with specified parameters.
        :param n_skills: Number of skills.
        :param min_skills_assigned: Minimal number of skills which can assigned to the task. It has to be greater than 0.
        :param max_skills_assigned: Maximal number of skills which task has to have. Smaller or equal than nSkills.
        If None, equals nSkills.
        :param difficulties_range: Tuple of minimum and maximum difficulties that task can have.
        :return: Random Task
        """
        if max_skills_assigned is None:
            max_skills_assigned = n_skills

        # number of skills in one task depends on exponential distribution with lambda = 0.8 scaled and rounded to [minSkill, maxSkill]
        skills_quantity_distribution = (
            np.random.exponential(10 / 8, 1)
            / 4
            * (max_skills_assigned - min_skills_assigned)
        )
        sampled_sized = np.floor(skills_quantity_distribution)[0]
        number_of_skills_assigned = min_skills_assigned + sampled_sized

        selected_skills = np.random.choice(
            np.arange(n_skills),
            replace=False,
            size=number_of_skills_assigned,
        )

        sampled_difficulties = np.random.random(selected_skills.shape[0])
        scaled_difficulties = sampled_difficulties * (
            difficulties_range[1] - difficulties_range[0]
        )
        scaled_difficulties += difficulties_range[0]

        task_to_difficulty = dict(zip(selected_skills, scaled_difficulties))

        return cls(task_difficulties=task_to_difficulty)

    def __str__(self):
        return f"Task difficulties : {str(list(self.task_difficulties.items()))}"
