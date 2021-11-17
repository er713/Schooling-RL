from .. import Task
from . import Teacher
from copy import deepcopy
from typing import List
import numpy as np


class BaseTeacher(Teacher):

    def __init__(self, nSkills: int, tasks: List[Task], estimateDifficulty: bool = True, **kwargs) -> None:
        copiedTasks = [deepcopy(task) for task in tasks]
        super().__init__(tasks=copiedTasks, nSkills=nSkills, **kwargs)
        self.studentsProficiencies = {}
        self.estimateDifficulty = estimateDifficulty

    def choose_task(self, student) -> Task:
        studentProficiencies = None
        if student.id in self.studentsProficiencies:
            studentProficiencies = self.studentsProficiencies[student.id]
        else:
            studentProficiencies = self.__get_new_student_proff()
            # add new student.id to studentProf dict
            self.studentsProficiencies[student.id] = studentProficiencies
        skillNum = np.argmin(studentProficiencies)
        studentProficiency = studentProficiencies[skillNum]

        # finding task with the most similar difficulty to student's efficiency
        min_diff = np.inf
        best_task = None
        for task in self.tasks:
            if skillNum in task.taskDifficulties:
                diff = abs(studentProficiency - task.taskDifficulties[skillNum])
                if diff < min_diff:
                    min_diff = diff
                    best_task = task
            else:
                continue
        return best_task

    def receive_result(self, result, reward=None, last=False) -> None:
        if result.task is None:
            return
        baseProfScaler = 0.1
        baseDiffScaler = 0.01
        unknownTask = False
        try:
            # if it was exam task reset record of student's proficiencies
            if result.isExam:
                self.studentsProficiencies[result.idStudent] = self.__get_new_student_proff()
                return
            studentProficiencies = self.studentsProficiencies[result.idStudent]
        except KeyError:
            raise KeyError(
                "StudentId: " + str(result.idStudent) + " hand out result, without asking for any task before.")
        mark = result.mark
        taskId = result.task.id
        # Check if result's task exists in teacher's tasks pool
        if result.task.id in [task.id for task in self.tasks]:
            estimatedTask = [task for task in self.tasks if task.id == taskId][0]
        else:
            unknownTask = True
            estimatedTask = result.task
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
            studentProficiencies[skillNo] += baseProfScaler * correctionScaler[skillNo]
            if self.estimateDifficulty and not unknownTask:
                estimatedTask.taskDifficulties[skillNo] -= baseDiffScaler * correctionScaler[skillNo]

    def __get_new_student_proff(self):
        return self.nSkills * [0]
      
    # def __str__(self):
    #     return 'BaseTeacher'
