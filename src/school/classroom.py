import random
from typing import Type, Tuple, Dict, List, Optional
import numpy as np

from . import Task, Result
from .students import Student
from .teachers import Teacher


class Classroom:
    def __init__(self, nSkills: int, teacherModel: Type[Teacher], studentModel: Type[Student],
                 difficultyRange: List[Optional[Dict[int, float]]] = None, nStudents: int = 1) -> None:
        """
        :param nSkills: The number of skills
        :param teacherModel: Class implementing Teacher interface.
        :param studentModel: Class implementing Student interface.
        :param difficultyRange: List of dictionaries of sill and difficulties. If None, generate Tasks (uniformly from
        -3 to 3 for each skill). If list of None, generate number of random Task equal to length of list.
        :param nStudents: The number of students in classroom
        """
        assert issubclass(teacherModel, Teacher)
        assert issubclass(studentModel, Student)
        if difficultyRange is None:
            difficultyRange = []
            for skill in range(nSkills):
                for difficulty in range(-3, 4):
                    difficultyRange.append({skill: difficulty})

        self.nSkills = nSkills
        self._studentModel = studentModel
        self.teacher = teacherModel()
        self.tasks = self._generate_tasks(difficultyRange)
        self.students = self._generate_students(nStudents)
        self._learning_types = {
            'single-student': self._learning_loop_single_student,
            'all-one-by-one': self._learning_loop_single_student,
            'all-random': self._learning_loop_all_student_parallel
        }

    def learning_process(self, timeToExam: int, learningType: str = 'all-one-by-one') -> List[Student]:
        """
        Function responsible for learning process
        :param timeToExam: Number of Tasks proposed to each Student.
        :param learningType: String, one of: 'single-student',' all-one-by-one', 'all-random'.
        :return: List of student who were taught.
        """
        assert learningType in self._learning_types
        return self._learning_types[learningType](timeToExam)

    def _learning_loop_single_student(self, timeToExam: int) -> List[Student]:
        """
        Learning loop for teaching one random Student.
        """
        student = self.students[np.random.randint(0, len(self.students), 1)[0]]
        for time in range(timeToExam):
            if student.want_task():
                self.teacher.receive_result(
                    student.solve_task(self.teacher.choose_task(student))
                )
        return [student]

    def _learning_loop_all_student(self, timeToExam: int) -> List[Student]:
        """
        Learning loop for teaching all students, one by one.
        """
        for student in self.students:
            for time in range(timeToExam):
                if student.want_task():
                    self.teacher.receive_result(
                        student.solve_task(self.teacher.choose_task(student), isExam=False))
        return self.students

    def _learning_loop_all_student_parallel(self, timeToExam: int) -> List[Student]:
        """
        Learning loop for teaching all students, random order.
        """
        for time in range(timeToExam):
            random.shuffle(self.students)
            for student in self.students:
                if student.want_task():
                    self.teacher.receive_result(
                        student.solve_task(self.teacher.choose_task(student)))
        return self.students

    def make_exam(self, examTasks: List[Task], students: List[Student]) -> (float, float):
        """
        Function responsible for evaluation process
        :param examTasks: List of Tasks.
        :param students: List of Student who taking the exam.
        """
        results = []
        for student in students:
            for task in examTasks:
                res = student.solve_task(task, True)
                self.teacher.receive_result(res)
                results.append(res)

        return Result.get_mean_result(results)

    def run(self, timeToExam: int, learningType: str = 'all-one-by-one', minimalImprove: Tuple[int, float] = None,
            minimalThreshold: Tuple[int, float] = None, numberOfIteration: int = 1,
            examTasksDifficulties: List[Optional[Dict[int, float]]] = None) -> None:
        """
        Function responsible for running learning and evaluation process
        :param timeToExam: Number of Tasks proposed to each Student.
        :param learningType: String, one of: 'single-student',' all-one-by-one', 'all-random'. For more check learning_loop.
        :param minimalImprove: Tuple of number of epoch in which improvement has to appear and how great it has to be
        to continue learning.
        :param minimalThreshold: Tuple of number of epoch in which threshold has to be exceeded to stop learning.
        Work only of minimalImprove is None.
        :param numberOfIteration: How many times learning_loop will be called.
        Works only if minimalImprove and minimalThreshold are None.
        :param examTasksDifficulties: List of dictionary of skill and difficulties. If None, generate 2 Task
        with difficulty 2 for each skill.
        """
        examTasks = []  # Generating tasks for exam
        if examTasksDifficulties is None:
            for skill in range(self.nSkills):
                for _ in range(2):
                    examTasks.append({skill: 2})
        else:
            examTasks = self._generate_tasks(examTasksDifficulties)

        if minimalImprove is not None:
            self._run_minimal_improvement(timeToExam, learningType, minimalImprove[0], minimalImprove[1], examTasks)
        elif minimalThreshold is not None:
            self._run_minimal_threshold(timeToExam, learningType, minimalThreshold[0], minimalThreshold[1], examTasks)
        else:
            for epoch in range(numberOfIteration):
                students = self.learning_process(timeToExam, learningType)
                result, _ = self.make_exam(examTasks, students)
                print(f"Epoch: {epoch}, mean score on exam: {result}")

    def _run_minimal_improvement(self, timeToExam: int, learningType: str, nEpoch: int, minImprovement: float,
                                 examTasks: List[Task]) -> None:
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
            students = self.learning_process(timeToExam, learningType)
            result, _ = self.make_exam(examTasks, students)
            print(f"Epoch: {epoch}, mean score on exam: {result}")
            if result - lastResult < minImprovement:
                epochWithoutImprovement += 1
            else:
                epochWithoutImprovement = 0
                lastResult = result
            if epochWithoutImprovement >= nEpoch:
                break

    def _run_minimal_threshold(self, timeToExam: int, learningType: str, nEpoch: int, threshold: float,
                               examTasks: List[Task]) -> None:
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
            students = self.learning_process(timeToExam, learningType)
            result, _ = self.make_exam(examTasks, students)
            print(f"Epoch: {epoch}, mean score on exam: {result}")
            if result >= threshold:
                epochWithThreshold += 1
            else:
                epochWithThreshold = 0
            if epochWithThreshold >= nEpoch:
                break

    def _generate_tasks(self, difficultyRange: List[Optional[Dict[int, float]]]) -> List[Task]:
        """
        Function responsible for generating tasks for Classroom
        :param difficultyRange: List of dictionaries with skill difficulties. If dictionary is None, generating
        random Task.
        """
        tasks = []
        for difficulties in difficultyRange:
            if difficulties is None:
                tasks.append(Task.generate_random_task(self.nSkills))
            else:
                tasks.append(Task(difficulties))
        return tasks

    def _generate_students(self, nStudents: int) -> List[Student]:
        """
        Function responsible for generating students for Classroom
        :param nStudents: number of students in Classroom
        """
        students = []
        for id_ in range(nStudents):
            students.append(
                self._studentModel(id_, list(np.clip(np.random.normal(scale=1 / 3, size=self.nSkills), -1, 1))))
        return students


if __name__ == "__main__":
    c = Classroom(7, Teacher, Student)
