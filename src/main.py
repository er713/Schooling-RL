"""
main file
"""
from school import Classroom, Plotter, import_results
from school.teachers import *
from school.students import RashStudent

if __name__ == '__main__':
   
    c = Classroom(7, TeacherActorCritic, RashStudent, nStudents=10, gamma=0.9, nLast=5, learning_rate=0.05)
    c.run(timeToExam=100, numberOfIteration=10, saveResults=False, visualiseResults=True, savePlot=False)

    # res = import_results('./data/RandomTeacher/RashStudent__100_7__2021-10-30_23-39.csv')
    # print(res[0].mark, res[0].isExam)

    # Plotter.plot_from_csv('./data/RandomTeacher/RashStudent__100_7__2021-10-31_2-40.csv',
    #                       './data/RandomTeacher/RashStudent__100_7__2021-10-31_2-40.png')
