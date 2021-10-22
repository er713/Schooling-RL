from .student import Student
from ..result import Result
from ..task import Task
from scipy.special import expit
import numpy as np
import random


class RashStudent(Student):

    def __init__(self, id: int, proficiency: list[float], desireToLearn: float = 1,
                 baseChangeParam: float = 0.1) -> None:
        """
        :param id: The id of student
        :param proficiency: The list of skills proficiency in range [-3,3].
        :param desireToLearn: The likelihood to do task [0,1]
        """
        super().__init__(id, proficiency, desireToLearn)
        self.id = id
        self._baseChangeParam = baseChangeParam
        self._proficiency = proficiency
        self._desireToLearn = desireToLearn

    def solve_task(self, task: Task, isExam: bool = False) -> Result:
        """
        Function resposible for solve task which triggers update proficiency
        Used formula for probability of correct answer p_corr=1/{1+e^[-(proficiency-difficulty)]}
        Final mark is mean of all probabilities for particular skills
        :param isExam: Is it exam task
        :param task: The Task object
        """
        probasToSolve = np.zeros(len(self._proficiency))
        for skill, difficulty in task.taskDifficulties.items():
            logit_p = self._proficiency[skill] - difficulty
            probasToSolve[skill] = expit(logit_p)
        probaToSolve = probasToSolve.mean(where=probasToSolve != 0)
        mark = 1 if random.random() < probaToSolve else 0
        taskResult = Result(mark, -1, task, self.id, isExam=isExam)
        if not isExam:
            self._update_proficiency(taskResult, probaToSolve)
        return taskResult

    def want_task(self) -> bool:
        """
        Function which return student choice to solve or not to solve a task,
        """
        return True if random.random() < self._desireToLearn else False

    def _update_proficiency(self, result: Result, probaToSolve: float = -1) -> None:
        """
        Function responsible for update student proficiency of given skill
        :param result: Result
        """
        diffs = np.zeros(len(self._proficiency))
        taskSkillsMask = np.zeros(len(self._proficiency), dtype=bool)
        for skill, difficulty in result.task.taskDifficulties.items():
            diffs[skill] = difficulty - self._proficiency[skill]
            taskSkillsMask[skill] = True
        # baseChange will be different from 0 only when task influence certain skill
        baseChange = np.zeros(len(self._proficiency))
        np.maximum(diffs, self._baseChangeParam, out=baseChange, where=taskSkillsMask)
        correctness_mod = 3 / 2 if result.mark == 1 else 2 / 3
        changeStrength = probaToSolve ** correctness_mod
        for idx, _ in enumerate(self._proficiency):
            self._proficiency[idx] += baseChange[idx] * changeStrength
