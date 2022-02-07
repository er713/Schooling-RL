from argparse import ArgumentParser
from itertools import product
from typing import Tuple, Dict

import gym
import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
from pl_bolts.models.rl import AdvantageActorCritic
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger

from school import Task
from school.students import RashStudent


class GradesBookEnvironment(Env):
    """
    As an action agent (teacher) can give a student one task. The task is related to some skill (ability to learn)
    and also difficulty. In the result number of possible action (tasks) is equal to quantity of possible
    difficulties multiplied by number of different skills (abilities).

    As an observation returned by the environment is something that can be interpreted as grade book.
    For each task given by teacher, the number of successes and number of fails is returned .
    So the resulting vector is two times longer than vector of possible actions. The range of values is from 0
    to number of possible fails (so simply the time of learning before the exam).

    Reward is calculated as follows:
    - 0 during first n-1 steps, student (our environment) has time to learn,
    - at n-th step exam is performed, so 2 tasks for each skill (of difficulty equal to 2) is given to student, reward
    at this moment is sum of solved tasks
    """

    POSSIBLE_TASK_DIFFICULTIES = [-3, -2, -1, 0, 1, 2, 3]

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
        self.observation_space = Box(
            low=0, high=time_to_exam, shape=(2 * self.number_of_tasks,), dtype=np.int
        )

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
        for action in tasks_todo:
            result = self.student.solve_task(self.tasks[action], isExam=True)
            is_task_solved = result.mark
            self.state[int(is_task_solved), action] += 1

            if is_task_solved and not is_learning_action:
                reward += 1

        info = {} if is_learning_action else {"exam_score": reward}
        self.iteration += 1

        return self.state.flatten(), reward, done, info

    def reset(self) -> np.array:
        student_proficiency = np.clip(
            np.random.normal(scale=1 / 3, size=self.skills_quantity), -1, 1
        )
        self.student = RashStudent(id=-1, proficiency=list(student_proficiency))
        self.state = np.zeros(shape=(2, self.number_of_tasks))
        self.iteration = 0
        return self.state.flatten()


if __name__ == "__main__":
    gym.envs.register(
        id="gradesbook-v0",
        entry_point="environments.grades_book:GradesBookEnvironment",
        kwargs={"skills_quantity": 1, "time_to_exam": 10},
    )

    parser = ArgumentParser(add_help=False)
    parser = Trainer.add_argparse_args(parser)
    parser = AdvantageActorCritic.add_model_specific_args(parser)
    args = parser.parse_args()

    seed_everything(123)
    model = AdvantageActorCritic(**args.__dict__)
    # wandb_logger = WandbLogger(project="schooling-rl", name="1 skill 10 tasks to exam")
    wandb_logger = None
    trainer = Trainer.from_argparse_args(args, deterministic=True, logger=wandb_logger)
    trainer.fit(model)
