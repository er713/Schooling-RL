from .. import Task
from . import Teacher
from copy import deepcopy
from typing import List
import numpy as np


class BaseTeacher(Teacher):

    def __init__(self, nSkills: int, tasks: List[Task], estimateDifficulty: bool = False, noExamTasks=None,
                 **kwargs) -> None:
        copiedTasks = [BaseTeacherTask(task) for task in tasks]
        # set all estimated difficulties to 0 if difficulties need to be estimated
        self.originalTasks = tasks
        if estimateDifficulty:
            for task in copiedTasks:
                for skillNo in task.taskDifficulties.keys():
                    task.taskDifficulties[skillNo] = np.inf
        super().__init__(tasks=copiedTasks, nSkills=nSkills, **kwargs)
        self.studentsNoSolvedTasks = {}
        self.estimateDifficulty = estimateDifficulty
        self.noExamTasks = noExamTasks
        self.noReceivedExamResults = {}

    def choose_task(self, student) -> Task:
        if student.id in self.studentsNoSolvedTasks:
            noSolvedTasks = self.studentsNoSolvedTasks[student.id]
        else:
            noSolvedTasks = self.__get_new_student_task_counter()
            # add new student.id to studentProf dict
            self.studentsNoSolvedTasks[student.id] = noSolvedTasks
        # find skill in which student trained the least
        skillNum = np.argmin(noSolvedTasks)
        # finding task with the highest difficulty for given skill
        max_diff = -np.inf
        best_task = None
        for task in self.tasks:
            if skillNum in task.taskDifficulties:
                curr_diff = task.taskDifficulties[skillNum]
                if curr_diff > max_diff:
                    max_diff = curr_diff
                    best_task = task
        self.choices[best_task.id] += 1
        original_best_task = [task for task in self.originalTasks if task.id == best_task.id][0]
        return original_best_task

    def receive_result(self, result, reward=None, last=None) -> None:
        if result.isExam:
            self.__receive_exam_result(result)
            return
        noSolvedTasks = self.studentsNoSolvedTasks[result.idStudent]
        for skillNo in result.task.taskDifficulties:
            noSolvedTasks[skillNo] += 1
        # select result task copy for difficulty estimation
        if self.estimateDifficulty:
            estimatedTask = [task for task in self.tasks if task.id == result.task.id][0]
            estimatedTask.update_task_estimation(result)

    def __get_new_student_task_counter(self):
        return self.nSkills * [int(0)]

    def __receive_exam_result(self, result):
        studentId = result.idStudent
        assert result.isExam
        if result.idStudent not in self.noReceivedExamResults:
            self.noReceivedExamResults[studentId] = 0
        self.noReceivedExamResults[studentId] += 1
        if self.noReceivedExamResults[studentId] == self.noExamTasks:
            self.noReceivedExamResults[studentId] = 0
            self.studentsNoSolvedTasks[studentId] = self.__get_new_student_task_counter()


class BaseTeacherTask(Task):
    def __init__(self, task):
        super().__init__(deepcopy(task.taskDifficulties), task.id)
        self.noTries = 0
        self.noSuccesses = 1

    def update_task_estimation(self, result):
        self.noTries += 1
        self.noSuccesses += result.mark
        if self.noTries >= 100:
            for skillNo in self.taskDifficulties.keys():
                self.taskDifficulties[skillNo] = self.noTries / (self.noTries + self.noSuccesses)
