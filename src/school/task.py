class Task:
    def __init__(self, id : int, taskDifficulties: dict) -> None:
        """
        :param difficulty: The difficulty of the task
        :param skill: The label of skill
        """
        self.id = id
        self.taskDifficulties = taskDifficulties
