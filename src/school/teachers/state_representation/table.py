from abc import abstractmethod
import tensorflow as tf
from typing import List
from ...teachers import TeacherRL
from ...task import Task
from ... import Result


class TableTeacher(TeacherRL):
    def __init__(self, nSkills: int, tasks: List[Task], timeToExam=None, noExamTasks=None, **kwargs):
        super().__init__(nSkills, tasks, **kwargs)
        self.students = {}  # students states
        self.students_inner = {}
        self.exam_results = {}
        self.exam_tasks = nSkills * 2
        self.time_to_exam = timeToExam

    def choose_task(self, student, is_learning: bool = True) -> Task:
        if student.id not in self.students:
            self.students[student.id] = self.create_new_student()
        return super().choose_task(student, is_learning)

    def receive_result(self, result, last=False, reward=None) -> None:
        if result.isExam:
            self.receive_exam_result(result)
            return
        self.update_memory(result, last)
        if last:
            # update epsilon
            super().receive_result(result)
            return
        # implementation specific learn

    def get_state(self, student):
        return tf.expand_dims(tf.constant(self.students[student.id]), axis=0)

    def receive_exam_result(self, result: Result):
        self.exam_results[result.idStudent].noAcquiredExamResults += 1
        self.exam_results[result.idStudent].marksSum += result.mark
        if self.exam_results[result.idStudent].noAcquiredExamResults == self.exam_tasks:
            self.update_memory(result, last=True)

    def create_new_student(self):
        # information about task consists of fraction of tries and fraction of successes
        return [0] * (2 * len(self.tasks))

    def update_student_state(self, result):
        """
            student_state=[ nTF_0, sF_0, nTF_1, sF_1, ...]
            _x - task number
            nTF_x - noTriesFraction = noTries/maxNoTries
            sF_x - successFraction = successes/noTries
            maxNoTries=timeToExam
            :param result: task result
            :return: updated student state
        """
        tries, successes = self.update_inner_student_state(result)
        if result.isExam:
            self.students.pop(result.idStudent)
            student_state = None
        else:
            student_state = self.students[result.idStudent]
            tries_fraction_idx = result.task.id * 2
            success_fraction_idx = tries_fraction_idx + 1
            student_state[tries_fraction_idx] = tries / self.time_to_exam
            student_state[success_fraction_idx] = successes / tries
        return student_state

    def update_inner_student_state(self, result) -> (int, int):
        """
                studentInnerState=[ ... numberOfTries_i, numberOfSuccesses_i ... ]
                :param result: task result
                :return: updated studentInnerState
        """
        tries = None
        successes = None
        if result.isExam:
            self.students_inner.pop(result.idStudent)
        else:
            if result.idStudent not in self.students_inner:
                self.students_inner[result.idStudent] = self.create_new_student()
            inner_student_state = self.students_inner[result.idStudent]
            noTryIdx = result.task.id * 2
            successes_idx = noTryIdx + 1
            inner_student_state[noTryIdx] += 1
            inner_student_state[successes_idx] += result.mark
            tries = inner_student_state[noTryIdx]
            successes = inner_student_state[successes_idx]
        return tries, successes
