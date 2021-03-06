import random
import warnings
from typing import Type, Tuple, Dict, List, Optional
import numpy as np
from os import makedirs
from datetime import datetime

from . import Task, Result, Plotter, import_results, export_results
from .students import Student
from .teachers import Teacher, TeacherNLastHistory


class Classroom:
    _SINGLE_STUDENT = 'single-student'
    _ALL_ONE_BY_ONE = 'all-one-by-one'
    _ALL_RANDOM = 'all-random'

    def __init__(self, nSkills: int, teacherModel: Type[Teacher], studentModel: Type[Student],
                 tasksSkillsDifficulties: List[Optional[Dict[int, float]]] = None, nStudents: int = 1,
                 minSkill: int = 1, maxSkill: int = None, difficultiesRange: Tuple[float, float] = (-3, 3),
                 saveResultsNumber: int = 1e5, now: datetime = None, **kwargs) -> None:
        """
        :param nSkills: The number of skills
        :param teacherModel: Class implementing Teacher interface.
        :param studentModel: Class implementing Student interface.
        :param tasksSkillsDifficulties: List of dictionaries of sill and difficulties. If None, generate Tasks
        (uniformly from -3 to 3 for each skill). If list of None, generate number of random Task equal to length of list.
        :param nStudents: The number of students in classroom
        :param minSkill: (Task generating) Minimal number of skill which task has to have. Greater than 0.
        :param maxSkill: (Task generating) Maximal number of skill which task has to have. Smaller or equal than nSkills.
        If None, equals nSkills.
        :param difficultiesRange: (Task generating) Tuple of minimum and maximum difficulties that task can have.
        :param saveResultsNumber: Number of Results for which cache is dumped to file (methods export_results).
        :param now: Datetime for creating name of file with exported Results.
        :param kwargs: Parameters for teacherModel constructor.
        """
        assert issubclass(teacherModel, Teacher)
        assert issubclass(studentModel, Student)

        Task.reset_index()  # For multiple sequential runs, e.g. on colab
        if tasksSkillsDifficulties is None:  # Default task, for each skill, difficulties in [-3, 3]
            tasksSkillsDifficulties = []
            for skill in range(nSkills):
                for difficulty in range(-3, 4):
                    tasksSkillsDifficulties.append({skill: difficulty})

        # Task generating parameters
        self._minSkill: int = minSkill
        self._maxSkill: int = maxSkill
        self._difficultiesRange: Tuple[float, float] = difficultiesRange

        # Initializing params
        self.nSkills: int = nSkills
        self.nStudents: int = nStudents
        self._studentModel: Type[Student] = studentModel

        self.students: List[Student] = []  # Generated during learning_process method
        self.tasks: List[Task] = self._generate_tasks(tasksSkillsDifficulties)

        self.teacher: Teacher = teacherModel(self.nSkills, self.tasks, noExamTasks=2 * self.nSkills,
                                             nStudents=nStudents, **kwargs)

        self.results: List[Result] = []
        self.saveTaskNumber: int = saveResultsNumber
        if now is None:
            self.now = datetime.now()
        else:
            self.now = now

        self._learning_types = {  # only for choosing method in learning_loop
            self._SINGLE_STUDENT: self._learning_loop_single_student,
            self._ALL_ONE_BY_ONE: self._learning_loop_all_student,
            self._ALL_RANDOM: self._learning_loop_all_student_parallel
        }

    def learning_process(self, timeToExam: int, learningType: str = 'all-one-by-one') -> None:
        """
        Function responsible for learning process
        :param timeToExam: Number of Tasks proposed to each Student.
        :param learningType: String, one of: 'single-student',' all-one-by-one', 'all-random'. Running specified
        _learning_loop method, respectively: _single_student, _all_student, _all_student_parallel.
        :return: List of student who were taught.
        """
        assert learningType in self._learning_types
        self.students = self._generate_students(self.nStudents)
        self._learning_types[learningType](timeToExam)

    def _learning_loop_single_student(self, timeToExam: int) -> None:
        """
        Learning loop for teaching one random Student.
        """
        student = self.students[np.random.randint(0, len(self.students), 1)[0]]
        for time in range(timeToExam):
            self._learn_student(student, last=(time == timeToExam - 1))

    def _learning_loop_all_student(self, timeToExam: int) -> None:
        """
        Learning loop for teaching all students, one by one.
        """
        for student in self.students:
            for time in range(timeToExam):
                self._learn_student(student, last=(time == timeToExam - 1))

    def _learning_loop_all_student_parallel(self, timeToExam: int) -> None:
        """
        Learning loop for teaching all students, random order.
        """
        for time in range(timeToExam):
            random.shuffle(self.students)
            for student in self.students:
                self._learn_student(student, last=(time == timeToExam - 1))

    def _learn_student(self, student: Student, isExam: bool = False, last: bool = False) -> None:
        """
        Assistant function for giving one Task to specified Student.
        :param student: Student to learn.
        :param isExam: If Task is part of exam.
        """
        if student.want_task():
            task = self.teacher.choose_task(student)
            result = student.solve_task(task, isExam=isExam)
            self.results.append(result)
            self.teacher.receive_result(result, last=last)

    def make_exam(self, examTasks: List[Task]) -> (float, float):
        """
        Function responsible for evaluation process
        :param examTasks: List of Tasks.
        """
        mean_results = []
        for student in self.students:
            results = []
            for task in examTasks:
                res = student.solve_task(task, True)
                self.results.append(res)
                self.teacher.receive_result(res)
                results.append(res)
            mean_mark, _ = Result.get_mean_result(results)
            mean_results.append(mean_mark)
            # Give teacher reward - mean result of one student on the exam
            if issubclass(self.teacher.__class__, TeacherNLastHistory):
                self.teacher.receive_result(Result(0, 0, None, idStudent=student.id, isExam=False), reward=mean_mark)

        return np.mean(mean_results), -1

    def run(self, timeToExam: int, learningType: str = 'all-one-by-one', minimalImprove: Tuple[int, float] = None,
            minimalThreshold: Tuple[int, float] = None, numberOfIteration: int = 1, visualiseResults: bool = True,
            examTasksDifficulties: List[Optional[Dict[int, float]]] = None, saveResults: bool = True,
            savePlot: bool = False) -> None:
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
        :param visualiseResults: Show graph with exam Results.
        :param examTasksDifficulties: List of dictionary of skill and difficulties. If None, generate 2 Task
        with difficulty 2 for each skill.
        :param saveResults: If True, dump all Results to file.
        :param savePlot: If True, save plot, works only if visualiseResults is True.
        """
        self.exportFileName: str = f"{self._studentModel.get_name()}__{self.nStudents}_{self.nSkills}__{timeToExam}__{numberOfIteration}__{self.now.year}-" + \
                                   f"{self.now.month}-{self.now.day}_{self.now.time().hour}-{self.now.time().minute}."

        if learningType == self._SINGLE_STUDENT and self.nStudents != 1:
            warnings.warn('Probably nStudents should be 1, because only one student will be taught before exam',
                          UserWarning)

        if visualiseResults:
            plotter = Plotter()
        else:
            plotter = None
        savePlot = savePlot and visualiseResults

        examTasks = []  # Generating tasks for exam
        if examTasksDifficulties is None:
            for skill in range(self.nSkills):
                for _ in range(2):
                    examTasks.append(Task({skill: 2}))
        else:
            examTasks = self._generate_tasks(examTasksDifficulties)

        if minimalImprove is not None:
            self._run_minimal_improvement(timeToExam, learningType, minimalImprove[0], minimalImprove[1], examTasks,
                                          saveResults=saveResults, plotter=plotter)
        elif minimalThreshold is not None:
            self._run_minimal_threshold(timeToExam, learningType, minimalThreshold[0], minimalThreshold[1], examTasks,
                                        saveResults=saveResults, plotter=plotter)
        else:
            for epoch in range(numberOfIteration):
                result, _ = self._run_single_learning(timeToExam, learningType, examTasks, saveResults, plotter)
                print(f"Epoch: {epoch}, mean score on exam: {result}")
        if savePlot:
            makedirs(f"./data/{self.teacher}/", exist_ok=True)
            plotter.plot_learning_save(f"./data/{self.teacher}/" + self.exportFileName + 'png')
            plotter.dump_results_to_csv(f"./data/{self.teacher}/" + self.exportFileName + 'csv')
    def _run_minimal_improvement(self, timeToExam: int, learningType: str, nEpoch: int, minImprovement: float,
                                 examTasks: List[Task], saveResults: bool = True, plotter: Plotter = None) -> None:
        """
        Function responsible for running learning until there is no improve.
        :param timeToExam: Frequency of evaluation, equal to learning_loop.
        :param learningType: String, one of: 'single-student',' all-one-by-one', 'all-random'. For more check learning_loop.
        :param nEpoch: Number of epochs when improvement has to appear to continue learning.
        :param minImprovement: Minimal difference to approve change as improvement.
        :param examTasks: List of Task on exam.
        :param saveResults: If True, dump all Results to file.
        :param plotter: Class for drawing plots.
        """
        lastResult = 0
        epoch = -1
        epochWithoutImprovement = 0

        while True:
            epoch += 1
            result, _ = self._run_single_learning(timeToExam, learningType, examTasks, saveResults, plotter)
            print(f"Epoch: {epoch}, mean score on exam: {result}")
            if result - lastResult < minImprovement:
                epochWithoutImprovement += 1
            else:
                epochWithoutImprovement = 0
                lastResult = result
            if epochWithoutImprovement >= nEpoch:
                break

    def _run_minimal_threshold(self, timeToExam: int, learningType: str, nEpoch: int, threshold: float,
                               examTasks: List[Task], saveResults: bool = True, plotter: Plotter = None) -> None:
        """
        Function responsible for running learning until score is above threshold for specified number of epoch.
        :param timeToExam: Frequency of evaluation, equal to learning_loop.
        :param learningType: String, one of: 'single-student',' all-one-by-one', 'all-random'. For more check learning_loop.
        :param nEpoch: Number of epochs for which threshold has to be exceeded to stop learning.
        :param threshold: Threshold.
        :param examTasks: List of Task on exam.
        :param saveResults: If True, dump all Results to file.
        :param plotter: Class for drawing plots.
        """
        epoch = -1
        epochWithThreshold = 0

        while True:
            epoch += 1
            result, _ = self._run_single_learning(timeToExam, learningType, examTasks, saveResults, plotter)
            print(f"Epoch: {epoch}, mean score on exam: {result}")
            if result >= threshold:
                epochWithThreshold += 1
            else:
                epochWithThreshold = 0
            if epochWithThreshold >= nEpoch:
                break

    def _run_single_learning(self, timeToExam: int, learningType: str, examTasks: List[Task],
                             saveResults: bool = True, plotter: Plotter = None) -> (float, float):
        """
        Assistant function for running one learning process.
        :param timeToExam: Frequency of evaluation, equal to learning_loop.
        :param learningType: String, one of: 'single-student',' all-one-by-one', 'all-random'. For more check learning_loop.
        :param examTasks: List of Task on exam.
        :param saveResults: If True, dump all Results to file.
        :param plotter: Class for drawing plots.
        :return: (mean mark on exam, mean duration on exam)
        """
        self.learning_process(timeToExam, learningType)
        mark, duration = self.make_exam(examTasks)
        if plotter is not None:
            plotter.plot_learning_results(mark)
        if saveResults:
            self.export_results(forceDump=True)  # TODO: think about doing this on other thread due to longer disk I/O
        return mark, duration

    def _generate_tasks(self, tasksSkillsDifficulties: List[Optional[Dict[int, float]]]) -> List[Task]:
        """
        Function responsible for generating tasks for Classroom
        :param tasksSkillsDifficulties: List of dictionaries with skill difficulties. If dictionary is None, generating
        random Task.
        :return: List of generated Tasks.
        """
        tasks = []
        for difficulties in tasksSkillsDifficulties:
            if difficulties is None:
                tasks.append(
                    Task.generate_random_task(self.nSkills, self._minSkill, self._maxSkill, self._difficultiesRange))
            else:
                tasks.append(Task(difficulties))
        return tasks

    def _generate_students(self, nStudents: int) -> List[Student]:
        """
        Function responsible for generating students for Classroom
        :param nStudents: number of students in Classroom
        :return: List of generated Students.
        """
        students = []
        for id_ in range(nStudents):
            students.append(
                self._studentModel(id_, list(np.clip(np.random.normal(scale=1 / 3, size=self.nSkills), -1, 1))))
        return students

    def export_results(self, path: str = None, forceDump: bool = False) -> None:
        """
        Function for exporting results only if cache exceeded specified size.
        :param path: If None, store results in data/str(teacher)/<parameters>_<date>.csv.
        Otherwise, store in path (relative from base path).
        :param forceDump: If True, omit condition check - always dump Results.
        """
        if len(self.results) == 0:
            return

        if forceDump or len(self.results) >= self.saveTaskNumber:
            if path is None:
                path = f"./data/{self.teacher}/"
            makedirs(path, exist_ok=True)  # creating directory if not exists

            export_results(path + self.exportFileName + 'csv', self.results)
            self.results = []

    def import_results(self, path: str = None, fileName: str = None) -> List[Result]:
        """
        Function for importing results
        :param path: If None, read results from data/str(teacher)/<parameters>_<date>.csv.
        Otherwise, read from path (relative from base path). Has to end with '/'.
        :param fileName: If None, set to exportFileName.
        :return: List of imported Results
        """
        if path is None:
            path = f"./data/{self.teacher}/"
        if fileName is None:
            fileName = self.exportFileName + 'csv'

        return import_results(path + fileName)


if __name__ == "__main__":
    c = Classroom(7, Teacher, Student)
    print(c)
