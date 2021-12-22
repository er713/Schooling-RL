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

    def _run_teacher(self, teacher, **kwargs):
        classroom = Classroom(2, teacher, MockStudent, nStudents=2, **kwargs)
        classroom.run(timeToExam=5, numberOfIteration=2, saveResults=False,
                      visualiseResults=False, savePlot=False)

    def test_random_teacher(self):
        self._run_teacher(RandomTeacher)

    def test_base_teacher(self):
        self._run_teacher(BaseTeacher)

    def test_teacher_ac_table(self):
        self._run_teacher(ActorCriticTableTeacher, cnn=False, timeToExam=5)

    def test_teacher_ac_nLast_cnn(self):
        self._run_teacher(ActorCriticNLastTeacher, cnn=True, nLast=3)

    def test_teacher_ac_all_history_cnn(self):
        self._run_teacher(ActorCriticAllHistoryCNNTeacher)

    def test_teacher_ac_all_history_rnn(self):
        self._run_teacher(ActorCriticAllHistoryRNNTeacher)

    def test_teacher_dqn_table(self):
        self._run_teacher(DQNTableTeacher, cnn=False, timeToExam=5)

    def test_teacher_dqn_nLast(self):
        self._run_teacher(DQNTeacherNLastHistory, cnn=False, nLast=3)

    def test_teacher_dqn_nLast_cnn(self):
        self._run_teacher(DQNTeacherNLastHistory, cnn=True, nLast=3)

    def test_teacher_dqn_all_history_cnn(self):
        self._run_teacher(DQNTeacherAllHistoryCNN)

    def test_teacher_dqn_all_history_rnn(self):
        self._run_teacher(DQNTeacherAllHistoryRNN)
