import unittest
from random import choice

from src.school import Classroom, Result, Task
from src.school.students import *
from src.school.teachers import Teacher


class MockTeacher(Teacher):

    def choose_task(self, student) -> Task:
        return choice(self.tasks)

    def receive_result(self, result) -> None:
        pass


class TestStudents(unittest.TestCase):
    """
    Tests for checking if Students do not give errors.
    """

    def _run_student(self, student):
        classroom = Classroom(5, MockTeacher, student, nStudents=2)
        classroom.run(timeToExam=10, numberOfIteration=10)

    def test_rash_student(self):
        self._run_student(RashStudent)
