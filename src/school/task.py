from __future__ import annotations
from typing import Dict, Tuple
import numpy as np


class Task:
    def __init__(self, taskDifficulties: Dict[int, float]) -> None:
        """
        :param taskDifficulties: The dictionary of difficulty for skill
        """
        self.id = next(Task._id_generator())
        self.taskDifficulties = taskDifficulties

    @classmethod
    def generate_random_task(cls, nSkills: int, minSkill: int = 1, maxSkill: int = None,
                             difficultiesRange: Tuple[float, float] = (-3, 3)) -> Task:
        """
        Create random Task with specified parameters.
        :param nSkills: Number of skills.
        :param minSkill: Minimal number of skill which task has to have. Greater than 0.
        :param maxSkill: Maximal number of skill which task has to have. Smaller or equal than nSkills.
        If None, equals nSkills.
        :param difficultiesRange: Tuple of minimum and maximum difficulties that task can have.
        :return: Random Task
        """
        if maxSkill is None:
            maxSkill = nSkills
        chooseNSkills = \
            np.random.choice(np.arange(nSkills), replace=False,
                             # number of skills in one task depends on exponential distribution
                             # with lambda = 0.8 scaled and rounded to [minSkill, maxSkill]
                             size=np.floor(np.random.exponential(10 / 8, 1) / 4 * (maxSkill - minSkill))[0] + minSkill)

        difficulties = np.random.random(chooseNSkills.shape[0]) * (difficultiesRange[1] - difficultiesRange[0]) + \
                       difficultiesRange[0]

        combined = dict()
        for skill, diff in zip(chooseNSkills, difficulties):
            combined[skill] = diff
        return cls(combined)

    @staticmethod
    def _id_generator():
        """
        ID generator, ensure distinct IDs.
        """
        _id = 0
        while True:
            yield _id
            _id += 1
