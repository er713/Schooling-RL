import random
from typing import List
from .teacher import Teacher
from .. import Task


class RandomTeacher(Teacher):
    def __init__(self, nSkills: int, tasks: List[Task] = None, **kwargs) -> None:
        if tasks is None:
            tasks = []
        super().__init__(nSkills, tasks=tasks, **kwargs)

    def choose_task(self, student) -> Task:
        action = random.choice(self.tasks)
        self.choices[action.id]
        return action

    def receive_result(self, result, reward=None, last=False) -> None:
        pass

    # def __str__(self):
    #     return 'RandomTeacher'
