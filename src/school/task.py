class Task:
    def __init__(self, difficulty: float, skill: int) -> None:
        """
        :param difficulty: The difficulty of the task
        :param skill: The label of skill
        """
        self.difficulty = difficulty
        self.skill = skill
