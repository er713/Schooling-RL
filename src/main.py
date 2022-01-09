"""
main file
"""
from school import Classroom, Plotter
from school.teachers import *
from school.students import RashStudent
import os
import argparse

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
            if '.csv' in file and 'tasks_' not in file:
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

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-iter', type=int, default=600, help="Number of iteration")
    parser.add_argument('-skills', type=int, default=1, help="Number of skills")
    parser.add_argument('-teacher', type=int, default=0, help="Choose teacher")
    parser.add_argument('-visualize', type=int, default=0, help="Visualize")
    # parser.add_argument('-verbose', type=int, default=0, help="Verbose")
    # parser.add_argument('-cnn', type=int, default=0, help="Use cnn? 0-1")

    return parser.parse_args()

if __name__ == '__main__':
    parsed_args = parseArguments()
    print(parsed_args)
    timeToExam = 6
    numberOfIteration = parsed_args.iter
    nSkills = parsed_args.skills
    teacher = BaseTeacher
    if parsed_args.teacher == 1:
        teacher = DQNTableTeacher
    elif parsed_args.teacher == 2:
        teacher = DQNTeacherNLastHistory
    elif parsed_args.teacher == 3:
        teacher = DQNTeacherAllHistoryRNN
    elif parsed_args.teacher == 4:
        teacher = ActorCriticTableTeacher
    elif parsed_args.teacher == 5:
        teacher = ActorCriticNLastTeacher
    elif parsed_args.teacher == 6:
        teacher = ActorCriticAllHistoryRNNTeacher
    c = Classroom(nSkills=nSkills,timeToExam=timeToExam*nSkills, teacherModel=teacher, studentModel=RashStudent, nStudents=100, verbose=True, cnn=False)
    if parsed_args.visualize:
        c.run(timeToExam=timeToExam*nSkills, numberOfIteration=numberOfIteration, saveResults=True, visualiseResults=True, savePlot=False)
    else:
        c.run(timeToExam=timeToExam*nSkills, numberOfIteration=numberOfIteration, saveResults=True, visualiseResults=False, savePlot=False)
