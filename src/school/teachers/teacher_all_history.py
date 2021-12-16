from typing import List
from . import TeacherNLastHistory
from .. import Task
from .layers import EmbeddedTasks
from abc import ABC
from .state_representation import get_state_history


class TeacherAllHistory(TeacherNLastHistory, ABC):
    def __init__(self, nSkills: int, tasks: List[Task], task_embedding_size: int = 5, base_history: int = 5, **kwargs):
        super().__init__(nSkills, tasks, None, None, **kwargs)

        self.task_embedding_size = task_embedding_size
        self.base_history = base_history
        self.embedding_for_tasks = EmbeddedTasks(self.nTasks, self.task_embedding_size, self.base_history)

    def get_state(self, student, shift=0):
        return get_state_history(self.results, student)
