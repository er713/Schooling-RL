class Result:
    def __init__(self, mark: float, duration: float, task: object, idStudent: int) -> None:
        """
        :param mark: result of task (for now 0 or 1 but in future it can continous [0,1])
        :param duration: how long it take to solve the task 
        :param task: task which has been done
        :param idStudent: id of student who done the task 
        """
        self.mark = mark
        self.duration = duration
        self.task = task
        self.idStudent = idStudent
