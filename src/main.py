"""
main file
"""
from school import Classroom, Plotter, import_results
from school.teachers import *
from school.students import RashStudent

import numpy as np

if __name__ == '__main__':
   
    c = Classroom(7, TeacherActorCritic, RashStudent, nStudents=50, gamma=0.99, nLast=5, learning_rate=0.05, verbose=False)
    c.run(timeToExam=30, numberOfIteration=15, saveResults=False, visualiseResults=True, savePlot=False)

    ch: np.ndarray = c.teacher.choices
    print(ch.mean(), ch.std())
    print(ch)
    # res = import_results('./data/RandomTeacher/RashStudent__100_7__2021-10-30_23-39.csv')
    # print(res[0].mark, res[0].isExam)

    # Plotter.plot_from_csv('./data/RandomTeacher/RashStudent__100_7__2021-10-31_2-40.csv',
    #                       './data/RandomTeacher/RashStudent__100_7__2021-10-31_2-40.png')
