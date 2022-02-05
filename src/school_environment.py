from argparse import ArgumentParser
from itertools import product

import gym
import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
from pl_bolts.models.rl import AdvantageActorCritic
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from school import Task
from school.students import RashStudent


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
                [number_of_not_solved_first_task, number_of_not_solved_second_task, ... ,
                number_of_solved_first_task , number of_solved_second_task, ... ]
            reward (float) : reward obtained in this iteration
            done (bool): whether the episode has ended
            info (dict): after epoch with exam specific information about score:
                'exam_score': int
        """
        self.iteration += 1

        is_learning_action = self.iteration // self.time_to_exam == 0
        done = not is_learning_action

        reward = 0
        tasks_todo = [action] if is_learning_action else self.test_task_ids
        for action in tasks_todo:
            result = self.student.solve_task(self.tasks[action], isExam=True)
            is_task_solved = result.mark
            self.state[int(is_task_solved), action] += 1

            if is_task_solved:
                reward += 1

        info = {} if is_learning_action else {"exam_score": reward}

        return self.state.flatten(), reward, done, info

    def combine_skill_and_difficulty_to_action(
        self, difficulty_id: int, skill_id: int
    ) -> int:
        return skill_id * self.number_of_difficulties + difficulty_id

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
        id="schoolenv-v0",
        entry_point="school_environment:SchoolEnv",
        kwargs={"skills_quantity": 5, "time_to_exam": 55},
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
    wandb_logger = WandbLogger(project="schooling-rl", name="5 skill 55 tasks to exam")
    trainer = Trainer.from_argparse_args(
        args, deterministic=True, callbacks=checkpoint_callback, logger=wandb_logger
    )
    trainer.fit(model)
