import random
from typing import List
from .teacher import Teacher
from .. import Task


class RandomTeacher(Teacher):
    def __init__(self, nSkills: int, tasks: List[Task] = None) -> None:
        if tasks is None:
            tasks = []
        super().__init__(nSkills, tasks=tasks)
    
    def choose_task(self, student) -> Task:
        return random.choice(self.tasks)

    def receive_result(self, result) -> None:
        pass
