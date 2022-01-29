from itertools import product
from random import random

from argparse import ArgumentParser
from typing import Iterator, Tuple

from torch import Tensor

import gym
import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
from pl_bolts.models.rl import AdvantageActorCritic
from pytorch_lightning import Trainer, seed_everything, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from scipy.special import expit


class SimpleRashStudent:
    def __init__(
        self,
        proficiency: np.array,
        desireToLearn: float = 1,
        baseChangeParam: float = 0.1,
    ) -> None:
        """
        :param proficiency: The list of skills proficiency in range [-3,3].
        :param desireToLearn: The likelihood to do task [0,1]
        :param baseChangeParam: TODO
        """
        self._baseChangeParam = baseChangeParam
        self._proficiency = proficiency
        self._desireToLearn = desireToLearn

    def solve_task(
        self, task_difficulty: int, skill_id: int, should_learn: bool
    ) -> bool:
        """
        Function responsible for solve task which triggers update proficiency
        Used formula for probability of correct answer p_corr=1/{1+e^[-(proficiency-difficulty)]}
        Final mark is mean of all probabilities for particular skills
        :param task_difficulty: TODO
        :param skill_id: TODO
        :param should_learn:
        """
        probability_to_solve_task = expit(self._proficiency[skill_id] - task_difficulty)
        is_solved = random() < probability_to_solve_task

        if should_learn:
            diff = task_difficulty - self._proficiency[skill_id]
            base_change = max(diff, self._baseChangeParam)

            correctness_mod = 3 / 2 if is_solved else 2 / 3
            change_strength = probability_to_solve_task ** correctness_mod

            self._proficiency[skill_id] += base_change * change_strength
            self._proficiency[skill_id] = np.clip(self._proficiency[skill_id], -3, 3)

        return is_solved

    def want_work(self) -> bool:
        return random() < self._desireToLearn


class SchoolEnv(Env):
    """
    As an action agent (teacher) can give a student one task. The task is related to some skill (ability to learn)
    and also difficulty. In the result number of possible action (tasks) is equal to quantity of possible
    difficulties multiplied by number of different skills (abilities).

    As an observation returned by the environment is something that can be interpreted as grade book.
    For each task (so the difficulty and skills as well) is returned number of successes and number of fails
    done by student. So the vector is two times longer than vector of possible actions. The range of values if from 0
    to number of possible fails (so simply the time of learning before the exam).

    Reward is calculated as follows:
    -1: for each failed task by student
    0: when student is procrastinating
    1: for each solved task by student

    The number of task done in one step:
    - during first n-1 steps student (our environment) has time to learn,
    - at n-th step exam is performed, so 2 tasks for each skill (of difficulty equal to 2) is given to student, reward
    at this moment is sum of solved tasks (without -1 points for not solved tasks)
    """

    POSSIBLE_TASK_DIFFICULTIES = [-3, -2, -1, 0, 1, 2, 3]

    def __init__(self, skills_quantity: int, time_to_exam: int):
        """
        :param skills_quantity: number of skills that can be learned
        :param time_to_exam: number of iterations before exam
        """
        self.skills_quantity = skills_quantity
        self.time_to_exam = time_to_exam
        self.reward_range = (-1, 2 * self.skills_quantity)

        self.number_of_difficulties = len(self.POSSIBLE_TASK_DIFFICULTIES)
        self.number_of_tasks = skills_quantity * self.number_of_difficulties
        self.action_space = Discrete(n=self.number_of_tasks)

        self.observation_space = Box(
            low=0, high=time_to_exam, shape=(2 * self.number_of_tasks,), dtype=np.int
        )

    def step(self, action: int):
        """
        Args:
            action (object): Task given to student, action space is constructed as follows:
            [first_skill_tasks, second_skill_tasks, ..., etc.] so for 3 skills and 7 difficulties we have 21 actions.
            action simply refers to trying to solve task of id 18 which means:
                skill of id 2 because 18 // 7 = 2 (action id divided by number of difficulties)
                difficulty of id 5 because 18 - (7 * 2) = 4

                Sanity check: last action of id 20 refers to difficulty of id 6 (so the last one when the number of
                skills is 7) as 20 - (2*7) = 6

                During epoch with exam epoch action is ignored

        Returns:
            observation np.array: vector of shape [1, 2*number of tasks] which is constructed as follows:
                [number_of_correctly_solved_first_task, number_of_incorrectly_solved_first_task,
                number_of_correctly_solved_second_task .. ]
                so value under the index 37 refers to number of not solved task number 18 because:
                37 // 2 = 18
                and 37 % 2 = 1 or 37 - (18 * 2) = 1
                Task number 18 refers to skill[2] and difficulty[5] as in action

            reward (float) : reward obtained in this iteration
            done (bool): whether the episode has ended
            info (dict): after epoch with exam specific information about score:
                'exam_score': int
        """
        self.iteration += 1

        reward = 0
        info = {}
        done = False

        is_learning_action = self.iteration // self.time_to_exam == 0

        if is_learning_action and self.student.want_work():
            skill_id, difficulty_id = self.extract_skill_difficulty_from_action(action)
            difficulty = self.POSSIBLE_TASK_DIFFICULTIES[difficulty_id]
            is_task_solved = self.student.solve_task(
                task_difficulty=difficulty, skill_id=skill_id, should_learn=True
            )
            # reward = 1 if is_task_solved else -1
            task_in_state_id = self.get_task_position_in_state(
                difficulty_id=difficulty_id, skill_id=skill_id, solved=is_task_solved
            )
            self.state[task_in_state_id] += 1

        else:
            for skill_id, _ in product(range(self.skills_quantity), range(2)):
                is_task_solved = self.student.solve_task(
                    task_difficulty=self.POSSIBLE_TASK_DIFFICULTIES[6],
                    skill_id=skill_id,
                    should_learn=False,
                )
                if is_task_solved:
                    reward += 1
                    task_in_state_id = self.get_task_position_in_state(
                        difficulty_id=6, skill_id=skill_id, solved=is_task_solved
                    )
                    self.state[task_in_state_id] += 1

            done = True
            info["exam_score"] = reward

        return self.state, reward, done, info

    def get_task_position_in_state(
        self, difficulty_id: int, skill_id: int, solved: bool
    ) -> int:
        return 2 * (skill_id * self.number_of_difficulties + difficulty_id) + int(
            not solved
        )

    def extract_skill_difficulty_from_action(self, action_id: int) -> Tuple[int, int]:
        """
        Takes as an input action_id (task_id) and returns skill_id and difficulty_id from that
        """
        skill_id = action_id // self.number_of_difficulties
        difficulty_id = action_id - skill_id * self.number_of_difficulties
        return skill_id, difficulty_id

    def reset(self) -> np.array:
        student_proficiency = np.clip(
            np.random.normal(scale=1 / 3, size=self.skills_quantity), -1, 1
        )
        self.student = SimpleRashStudent(proficiency=student_proficiency)
        self.state = np.zeros(shape=(2 * self.number_of_tasks))
        self.iteration = 0
        return self.state


if __name__ == "__main__":
    gym.envs.register(
        id="schoolenv-v0",
        entry_point="school_environment:SchoolEnv",
        kwargs={"skills_quantity": 2, "time_to_exam": 20},
    )

    parser = ArgumentParser(add_help=False)
    parser = Trainer.add_argparse_args(parser)
    parser = AdvantageActorCritic.add_model_specific_args(parser)
    args = parser.parse_args()

    model = AdvantageActorCritic(**args.__dict__)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1, monitor="avg_reward", mode="max", verbose=True
    )

    seed_everything(123)
    wandb_logger = WandbLogger(project="schooling-rl", name="2 skill 20 tasks to exam")
    trainer = Trainer.from_argparse_args(
        args, deterministic=True, callbacks=checkpoint_callback, logger=wandb_logger
    )
    trainer.fit(model)
