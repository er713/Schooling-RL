from abc import abstractmethod
from .. import Task
from teacher import Teacher
from copy import deepcopy
import numpy as np


class BaseTeacher(Teacher):

    def __init__(self, tasks: list[Task], nSkills: int, estimateDifficulty: bool) -> None:
        copiedTasks = [deepcopy(task) for task in tasks]
        super().__init__(tasks=copiedTasks, nSkills=nSkills)
        self.studentsProficiencies = {}
        self.estimateDifficulty = estimateDifficulty

    def choose_task(self, student) -> Task:
        studentProficiencies = None
        if student.id in self.studentsProficiencies:
            studentProficiencies = self.studentsProficiencies[student.id]
        else:
            studentProficiencies = [0]*self.nSkills
            # add new student.id to studentProf dict
            self.studentsProficiencies[student.id] = studentProficiencies
        skillNum = np.argmin(studentProficiencies)
        studentProficiency = studentProficiencies[skillNum]

        # finding task with the most similar difficulty to student's efficiency
        min_diff = np.inf
        best_task = None
        for task in self.tasks:
            diff = abs(studentProficiency - task.taskDifficulties[skillNum])
            if diff < min_diff:
                min_diff = diff
                best_task = task

        return best_task

    def receive_result(self, result) -> None:
        baseProfScaler = 0.1
        baseDiffScaler = 0.01
        studentProficiencies = self.studentsProficiencies[result.idStudent]
        mark = result.mark
        taskId = result.task.id
        estimatedTask = [task for task in self.tasks if task.id == taskId][0]
        relativeDifficulties = {}
        # calculate relative difficulties
        for skillNo, difficulty in estimatedTask.taskDifficulties.items():
            relativeDifficulties[skillNo] = difficulty - studentProficiencies[skillNo]
        correctionScaler = {}
        # update estimate of student's proficiency and difficulty of task's copy if needed
        for skillNo, relativeDifficulty in relativeDifficulties.items():
            if mark == 1:
                correctionScaler[skillNo] = 1 + max([relativeDifficulty, -1])
            else:
                correctionScaler[skillNo] = -(1 - min([relativeDifficulty, 1]))
            studentProficiencies[skillNo] += baseProfScaler*correctionScaler[skillNo]
            if self.estimateDifficulty:
                estimatedTask.taskDifficulties[skillNo] -= baseDiffScaler*correctionScaler[skillNo]

