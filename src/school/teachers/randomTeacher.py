import random
from .teacher import Teacher
from .. import Task


class RandomTeacher(Teacher):
    def __init__(self, tasks: list[Task] = []) -> None:
        super().__init__(tasks=tasks)
    
    def choose_task(self, student) -> Task:
        return random.choice(self.tasks)

    def receive_result(self, result) -> None:
        pass