class Classroom:
    def __init__(
        self,
        nSkills: int,
        difficultyRange: list = [-3, -2, -1, 0, 1, 2, 3],
        nStudents: int = 1,
    ) -> None:
        """
        :param nSkills: The number of skills
        :param difficultyRange: The range of tasks difficulty.
        :param nStudents: The number of students in classroom
        """
        self.teacher
        self.tasks = []
        self.students = []
        self.nSkils = nSkills

    def learning_process(self, timeToExam: int) -> None:
        """
        Function resposible for learning proccess
        :param timeToExam: frequency of evelauation
        """
        raise NotImplementedError("choose_task was not implemented")

    def make_exam(self) -> None:
        """
        Function resposible for evaluation process
        """
        raise NotImplementedError("make_exam was not implemented")

    def run(self) -> None:
        """
        Function resposible for runing learning and evalutaion process
        """
        raise NotImplementedError("run was not implemented")

    def _generate_tasks(difficultyRange: list, nSkills: int) -> None:
        """
        Function responsible for generating tasks for Classroom
        :param difficultyRange: difficulty range for task
        :param nSkills: number of skill that student can learn
        """
        raise NotImplementedError("_generate_tasks was not implemented")

    def _generate_students(self, nStudents: int) -> None:
        """
        Function resposible for generating students for Classroom
        :param nStudents: number of students in Classroom
        """
        raise NotImplementedError("_generate_students was not implemented")
