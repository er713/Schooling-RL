import unittest
from random import random

from src.school import Classroom, Result, Task
from src.school.teachers import *
from src.school.students import Student


class MockStudent(Student):

    def solve_task(self, task: Task, isExam: bool = False) -> Result:
        return Result(random(), random(), task, self.id, isExam)

    def want_task(self) -> bool:
        return True

    def _update_proficiency(self, result: Result) -> None:
        pass


class TestTeachers(unittest.TestCase):
    """
    Tests for checking if Teachers do not give errors.
    """

    def _run_teacher(self, teacher):
        classroom = Classroom(5, teacher, MockStudent, nStudents=2)
        classroom.run(timeToExam=10, numberOfIteration=10)

    def test_random_teacher(self):
        self._run_teacher(RandomTeacher)

    def test_base_teacher(self):
        self._run_teacher(BaseTeacher)
