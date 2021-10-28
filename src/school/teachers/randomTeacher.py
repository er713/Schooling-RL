import random
from .teacher import Teacher
from .. import Task


class RandomTeacher(Teacher):
    def __init__(self, nSkills: int, tasks: list[Task] = None) -> None:
        if tasks is None:
            tasks = []
        super().__init__(nSkills, tasks=tasks)
    
    def choose_task(self, student) -> Task:
        return random.choice(self.tasks)

    def receive_result(self, result) -> None:
        pass

    def __str__(self):
        return 'RandomTeacher'
