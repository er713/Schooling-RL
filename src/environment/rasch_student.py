from scipy.special import expit
from typing import List
import numpy as np
import random

from environment.task import Task


class RaschStudent:
    def __init__(
        self,
        proficiency: List[float],
        desire_to_learn: float = 1,
        base_change_param: float = 0.1,
    ) -> None:
        """
        :param proficiency: The list of skills proficiency in range [-3,3].
        :param desire_to_learn: The likelihood to do task [0,1]
        """
        self._proficiency = proficiency
        self._desire_to_learn = desire_to_learn
        self._base_change_param = base_change_param
        assert (
            self._base_change_param > 0
        ), "base_change_param can not be lower or equal to 0"

    def solve_task(self, task: Task, is_exam: bool = False) -> bool:
        """
        Function responsible for solve task which triggers update proficiency
        Used formula for probability of correct answer p_corr=1/{1+e^[-(proficiency-difficulty)]}
        Final mark is mean of all probabilities for particular skills
        :param is_exam: Is it exam task
        :param task: The Task object
        """
        probability_to_solve_task = np.zeros(len(self._proficiency))
        for skill, difficulty in task.task_difficulties.items():
            logit_p = self._proficiency[skill] - difficulty
            probability_to_solve_task[skill] = expit(logit_p)
        probability_to_solve_task = np.mean(
            probability_to_solve_task[probability_to_solve_task != 0]
        )
        is_task_solved = random.random() < probability_to_solve_task
        if not is_exam:
            self._update_proficiency(
                is_task_solved, task, float(probability_to_solve_task)
            )
        return is_task_solved

    def want_task(self) -> bool:
        """
        Function which return student choice to solve or not to solve a task,
        """
        return random.random() < self._desire_to_learn

    def _update_proficiency(
        self, is_task_solved: bool, task: Task, probability_to_solve_task: float
    ) -> None:
        """
        Function responsible for update student proficiency of given skill
        """
        proficiency_difference = np.zeros(len(self._proficiency))
        skills_in_task_mask = np.zeros(len(self._proficiency), dtype=bool)

        for skill, difficulty in task.task_difficulties.items():
            proficiency_difference[skill] = difficulty - self._proficiency[skill]
            skills_in_task_mask[skill] = True

        # base_change will be different from 0 only when task influence certain skill
        base_change = np.zeros(len(self._proficiency))
        np.maximum(
            proficiency_difference,
            self._base_change_param,
            out=base_change,
            where=skills_in_task_mask,
        )

        correctness_mod = 3 / 2 if is_task_solved else 2 / 3
        change_strength = probability_to_solve_task ** correctness_mod
        for idx in range(len(self._proficiency)):
            self._proficiency[idx] += base_change[idx] * change_strength

        self._proficiency = np.clip(self._proficiency, -3, 3)

    @staticmethod
    def get_name():
        return "RashStudent"
