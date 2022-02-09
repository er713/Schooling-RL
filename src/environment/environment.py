from abc import ABC
from itertools import product
from typing import Tuple, Dict

import numpy as np
from gym import Env
from gym.spaces import Discrete, Box

from environment.task import Task
from environment.rasch_student import RaschStudent


class BaseSchoolEnvironment(Env, ABC):
    """
    As an action agent (teacher) can give a student one task. The task is related to some skill (ability to learn)
    and also difficulty. In the result number of possible action (tasks) is equal to quantity of possible
    difficulties multiplied by number of different skills (abilities).

    As an observation depends on used grades evidence model.

    Reward is calculated as follows:
    - 0 during first n-1 steps, student (our environment) has time to learn,
    - at n-th step exam is performed, so 2 tasks for each skill (of difficulty equal to 2) is given to student, reward
    at this moment is sum of solved tasks
    """

    POSSIBLE_TASK_DIFFICULTIES = [-3, -2, -1, 0, 1, 2, 3]
    state: np.array

    def __init__(self, skills_quantity: int, time_to_exam: int):
        """
        :param skills_quantity: number of skills that can be learned
        :param time_to_exam: number of iterations before exam
        """
        self.skills_quantity = skills_quantity
        self.time_to_exam = time_to_exam
        self.reward_range = (0, 2 * self.skills_quantity)

        self.tasks = []
        self.test_task_ids = []
        for task_id, (difficulty, skill_id) in enumerate(
            product(self.POSSIBLE_TASK_DIFFICULTIES, range(skills_quantity))
        ):
            self.tasks.append(Task({skill_id: difficulty}))
            if difficulty == 2:
                self.test_task_ids.append(task_id)
                self.test_task_ids.append(task_id)

        self.number_of_difficulties = len(self.POSSIBLE_TASK_DIFFICULTIES)
        self.number_of_tasks = len(self.tasks)

        self.action_space = Discrete(n=self.number_of_tasks)

    def step(self, action: int) -> Tuple[np.array, float, bool, Dict]:
        """
        :param action: Task given to student. During epoch with exam epoch action is ignored
        :return:
            observation (np.array): vector of shape [1, 2*number of tasks] which is constructed as follows:
                [number_of_not_solved_first_task, number_of_not_solved_second_task, ... ,
                number_of_solved_first_task , number of_solved_second_task, ... ]
            reward (float) : reward obtained in this iteration
            done (bool): whether the episode has ended
            info (dict): after epoch with exam specific information about score:
                'exam_score': int
        """

        is_learning_action = (self.iteration + 1) // self.time_to_exam == 0
        done = not is_learning_action

        reward = 0
        tasks_todo = [action] if is_learning_action else self.test_task_ids
        for i, action in enumerate(tasks_todo):
            result = self.student.solve_task(
                self.tasks[action], isExam=not is_learning_action
            )
            is_task_solved = result.mark
            self.update_state(
                is_task_solved=bool(is_task_solved), task_id=action, task_in_epoch_id=i
            )

            if is_task_solved and not is_learning_action:
                reward += 1

        info = {}
        if done:
            info = {
                "exam_score": reward,
                "exam_score_percentage": reward / self.number_of_tasks,
            }

        self.iteration += 1

        return self.state.flatten(), reward, done, info

    def reset(self) -> np.array:
        student_proficiency = np.clip(
            np.random.normal(scale=1 / 3, size=self.skills_quantity), -1, 1
        )
        self.student = RaschStudent(id=-1, proficiency=list(student_proficiency))
        self.iteration = 0
        self.reset_state()
        return self.state.flatten()

    def update_state(
        self, is_task_solved: bool, task_id: int, task_in_epoch_id: int
    ) -> None:
        raise NotImplementedError

    def reset_state(self) -> None:
        raise NotImplementedError


class GradesBookEnvironment(BaseSchoolEnvironment):
    """
    As an observation returned by the environment is something that can be interpreted as grade book.
    For each task given by teacher, the number of successes and number of fails is returned .
    So the resulting vector is two times longer than vector of possible actions. The range of values is from 0
    to number of possible fails (so simply the time of learning before the exam).
    """

    def __init__(self, skills_quantity: int, time_to_exam: int):
        """
        :param skills_quantity: number of skills that can be learned
        :param time_to_exam: number of iterations before exam
        """

        super().__init__(skills_quantity, time_to_exam)
        self.observation_space = Box(
            low=0, high=time_to_exam, shape=(2 * self.number_of_tasks,), dtype=np.int
        )

    def update_state(
        self, is_task_solved: bool, task_id: int, task_in_epoch_id: int
    ) -> None:
        self.state[int(is_task_solved), task_id] += 1

    def reset_state(self) -> None:
        self.state = np.zeros(shape=(2, self.number_of_tasks))


class GradesListEnvironment(BaseSchoolEnvironment):
    """
    As an observation returned by the environment is something that can be interpreted as grades list.
    For each task given by teacher, the student's result is returned together with one-hot-encoded.
    [task_1 solved or not, one hot task_1 id, # timestep at t = 0
    [task_2 solved or not, one hot task_2 id, # timestep at t = 1
    ... etc]

    So the resulting vector is has length:
    (learning time + number of exam tasks) * (1 + number of tasks) i.e
    (n - 1 + number_of_skills * 2 ) * ( 1 + number_of_skills * number_of_difficulties)
    """

    def __init__(self, skills_quantity: int, time_to_exam: int):
        """
        :param skills_quantity: number of skills that can be learned
        :param time_to_exam: number of iterations before exam
        """
        super().__init__(skills_quantity, time_to_exam)
        self.tasks_todo_in_epoch = self.time_to_exam - 1 + self.skills_quantity * 2
        self.one_task_encoding = 1 + self.number_of_tasks
        self.observation_space = Box(
            low=0,
            high=1,
            shape=(self.tasks_todo_in_epoch * self.one_task_encoding,),
            dtype=np.int,
        )

    def update_state(
        self, is_task_solved: bool, task_id: int, task_in_epoch_id: int
    ) -> None:
        self.state[self.iteration + task_in_epoch_id, 0] = int(is_task_solved)
        self.state[self.iteration + task_in_epoch_id, task_id + 1] = 1

    def reset_state(self) -> None:
        self.state = np.zeros(shape=(self.tasks_todo_in_epoch, self.one_task_encoding))
