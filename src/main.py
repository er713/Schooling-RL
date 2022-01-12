"""
main file
"""
from school import Classroom, Plotter, import_results
from school.teachers import *
from school.students import RashStudent
import os
from school import Plotter
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
    timeToExam = 5
    nSkills = 1
    c = Classroom(1, DQNTeacherNLastHistory, RashStudent, nLast=4, nStudents=100, estimateDifficulty=False)
    # path = './data/table/'
    # files = [
    #     path+'1.csv',
    #     path+'2.csv',
    #     path+'3.csv', 
    #     path+'4.csv',
    #     path+'5.csv'

    #     ]
    # for i in range(1,5):
    #     Plotter.draw_tasks_distribution(path+f'tasks_skill_1_no_{i}.csv')
    # Plotter.plot_from_csv_with_std(files,path+'dqn-table-2.jpg', title='DQNTable-1Skills')
    # for i in range(10):
    #     c = Classroom(nSkills=nSkills,
    #                 teacherModel=RandomTeacher,
    #                 studentModel=RashStudent,
    #                 timeToExam=timeToExam,
    #                 nStudents=100,
    #                 gamma=0.99,
    #                 epsilon=0.9,
    #                 decay_epsilon=0.9992,
    #                 learning_rate=0.05,
    #                 min_eps=0.03,
    #                 verbose=True,
    #                 cnn=False)
    c.run(timeToExam=timeToExam, numberOfIteration=33, saveResults=True,
        visualiseResults=True, savePlot=False)
    
    # create_base_line()
    # res = import_results('./data/RandomTeacher/RashStudent__100_7__2021-10-30_23-39.csv')
    # print(res[0].mark, res[0].isExam)

    # Plotter.plot_from_csv('./data/RandomTeacher/RashStudent__100_7__2021-11-26_22-4.csv',
    #                       './data/RandomTeacher/RashStudent__100_7__2021-10-31_2-40.png')
