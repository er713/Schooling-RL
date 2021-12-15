"""
main file
"""
from school import Classroom, Plotter, import_results
from school.teachers import *
from school.students import RashStudent
import os

PATH_BASE = './data/BaseTeacher'
PATH_RANDOM = './data/RandomTeacher'
SKILLS = [1, 5, 10, 15, 20, 25]


def count_base_lines():
    for teacher in [RandomTeacher, BaseTeacher]:
        for skills in SKILLS:
            for tasks in [10, 15, 20]:
                c = Classroom(skills, teacher, RashStudent, nStudents=100, estimateDifficulty=False)
                c.run(timeToExam=skills * tasks, numberOfIteration=10, saveResults=True, visualiseResults=False,
                      savePlot=False)


def draw_all_plots_separately():
    for path in [PATH_BASE, PATH_RANDOM]:
        files = os.listdir(path)
        for file in files:
            if '.csv' in file:
                Plotter.plot_from_csv(f'{path}/{file}', f'{path}/{file.replace("csv", "png")}')


def draw_all_on_one_by_skills():
    paths = []
    for path in [PATH_BASE, PATH_RANDOM]:
        files = os.listdir(path)
        for file in files:
            if 'csv' in file:
                paths.append(f'{path}/{file}')

    for skills in SKILLS:
        Plotter.plot_multiple_from_csv(paths, f"./data/combine-{skills}.png", nSkills=skills)


def create_base_line():
    count_base_lines()
    draw_all_on_one_by_skills()
    draw_all_plots_separately()


if __name__ == '__main__':
    timeToExam = 10
    nSkills = 1
    # c = Classroom(1, RandomTeacher, RashStudent, nStudents=100, estimateDifficulty=False)

    c = Classroom(nSkills=1,
                  teacherModel=ActorCriticAllHistoryCNNTeacher,
                  studentModel=RashStudent,
                  timeToExam=timeToExam,
                  nStudents=100,
                  gamma=0.99,
                  epsilon=0.9,
                  decay_epsilon=0.999,
                  learning_rate=0.05,
                  verbose=True,
                  cnn=True)
    c.run(timeToExam=timeToExam, numberOfIteration=1000, saveResults=True, visualiseResults=True, savePlot=True)

    # create_base_line()
    # res = import_results('./data/RandomTeacher/RashStudent__100_7__2021-10-30_23-39.csv')
    # print(res[0].mark, res[0].isExam)

    # Plotter.plot_from_csv('./data/RandomTeacher/RashStudent__100_7__2021-11-26_22-4.csv',
    #                       './data/RandomTeacher/RashStudent__100_7__2021-10-31_2-40.png')
