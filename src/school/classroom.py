from typing import Type, Tuple
import numpy as np

from src.school.students.student import Student
from src.school.task import Task
from src.school.teachers.result import Result
from src.school.teachers.teacher import Teacher


class Classroom:
    def __init__(self, nSkills: int, teacherModel: Type[Teacher], studentModel: Type[Student], difficultyRange=None,
                 nStudents: int = 1) -> None:
        """
        :param nSkills: The number of skills
        :param teacherModel: Class implementing Teacher interface.
        :param studentModel: Class implementing Student interface.
        :param difficultyRange: The range of tasks difficulty.
        :param nStudents: The number of students in classroom
        """
        assert issubclass(teacherModel, Teacher)
        assert issubclass(studentModel, Student)
        if difficultyRange is None:
            difficultyRange = [-3, -2, -1, 0, 1, 2, 3]

        self.nSkills = nSkills
        self._studentModel = studentModel
        self.teacher = teacherModel()
        self._generate_tasks(difficultyRange, nSkills)
        self._generate_students(nStudents)

    def learning_process(self, timeToExam: int) -> None:
        """
        Function responsible for learning process
        :param timeToExam: frequency of evaluation
        """
        for student in self.students:
            for time in range(timeToExam):
                if student.want_task():
                    self.teacher.receive_result(
                        student.solve_task(self.teacher.choose_task(student)))

    def make_exam(self, taskForSkill: int = 2, tasksDifficulty: int = 2) -> (float, float):
        """
        Function responsible for evaluation process
        :param taskForSkill: Number of task in exam for one skill.
        :param tasksDifficulty: Difficulty of tasks on exam.
        """
        exam_tasks = [task for task in self.tasks if task.difficulty == tasksDifficulty]
        assert len(exam_tasks) == self.nSkills  # Make sure there is one task for one skill.
        results = []
        for student in self.students:
            results_student = []

            for task in exam_tasks:
                for _ in range(taskForSkill):
                    res = student.solve_task(task)
                    self.teacher.receive_result(res)
                    results_student.append(res)

            [results.append(r) for r in results_student]
        return Result.get_mean_result(results)

    def run(self, timeToExam: int, minimalImprove: Tuple[int, float] = None, minimalThreshold: Tuple[int, float] = None,
            numberOfIteration: int = 1) -> None:
        """
        Function responsible for running learning and evaluation process
        :param timeToExam: Frequency of evaluation, equal to learning_loop.
        :param minimalImprove: Tuple of number of epoch in which improvement has to appear and how great it has to be to continue learning.
        :param minimalThreshold: Tuple of number of epoch in which threshold has to be exceeded to stop learning. Work only of minimalImprove is None.
        :param numberOfIteration: How many times learning_loop will be called. Works only if minimalImprove and minimalThreshold are None.
        """
        if minimalImprove is not None:
            self._run_minimal_improvement(timeToExam, minimalImprove[0], minimalImprove[1])
        elif minimalThreshold is not None:
            self._run_minimal_threshold(timeToExam, minimalThreshold[0], minimalThreshold[1])
        else:
            for epoch in range(numberOfIteration):
                self.learning_process(timeToExam)
                result, _ = self.make_exam()
                print(f"Epoch: {epoch}, mean score on exam: {result}")

    def _run_minimal_improvement(self, timeToExam: int, nEpoch: int, minImprovement: float) -> None:
        """
        Function responsible for running learning until there is no improve.
        :param timeToExam: Frequency of evaluation, equal to learning_loop.
        :param nEpoch:
        :param minImprovement: Tuple of number of epoch in which improvement has to appear and how great it has to be to continue learning.
        """
        lastResult = 0
        epoch = -1
        epochWithoutImprovement = 0

        while True:
            epoch += 1
            self.learning_process(timeToExam)
            result, _ = self.make_exam()
            print(f"Epoch: {epoch}, mean score on exam: {result}")
            if result - lastResult < minImprovement:
                epochWithoutImprovement += 1
            else:
                epochWithoutImprovement = 0
                lastResult = result
            if epochWithoutImprovement >= nEpoch:
                break

    def _run_minimal_threshold(self, timeToExam: int, nEpoch: int, threshold: float) -> None:
        """
        Function responsible for running learning until score is above threshold for specified number of epoch.
        :param timeToExam: Frequency of evaluation, equal to learning_loop.
        :param nEpoch: 
        :param threshold: Tuple of number of epoch in which threshold has to be exceeded to stop learning.
        """
        epoch = -1
        epochWithThreshold = 0

        while True:
            epoch += 1
            self.learning_process(timeToExam)
            result, _ = self.make_exam()
            print(f"Epoch: {epoch}, mean score on exam: {result}")
            if result >= threshold:
                epochWithThreshold += 1
            else:
                epochWithThreshold = 0
            if epochWithThreshold >= nEpoch:
                break

    def _generate_tasks(self, difficultyRange: list, nSkills: int) -> None:
        """
        Function responsible for generating tasks for Classroom
        :param difficultyRange: difficulty range for task
        :param nSkills: number of skill that student can learn
        """
        self.tasks = []
        [[self.tasks.append(Task(diff, skill)) for diff in difficultyRange] for skill in range(nSkills)]

    def _generate_students(self, nStudents: int) -> None:
        """
        Function responsible for generating students for Classroom
        :param nStudents: number of students in Classroom
        """
        self.students = []
        for id_ in range(nStudents):
            self.students.append(self._studentModel(id_, np.random.random(self.nSkills) * 2 - 1))


if __name__ == "__main__":
    c = Classroom(7, Teacher, Student)
