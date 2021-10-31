"""
main file
"""
from school import Classroom, Plotter, import_results
from school.teachers import RandomTeacher, BaseTeacher
from school.students import RashStudent

if __name__ == '__main__':
    c = Classroom(7, RandomTeacher, RashStudent, nStudents=100)
    c.run(timeToExam=100, numberOfIteration=10, saveResults=True, visualiseResults=False, savePlot=True)

    # res = import_results('./data/RandomTeacher/RashStudent__100_7__2021-10-30_23-39.csv')
    # print(res[0].mark, res[0].isExam)

    # Plotter.plot_from_csv('./data/RandomTeacher/RashStudent__100_7__2021-10-31_2-40.csv',
    #                       './data/RandomTeacher/RashStudent__100_7__2021-10-31_2-40.png')
