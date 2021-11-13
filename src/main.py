"""
main file
"""
from school import Classroom, Plotter, import_results
from school.teachers import *
from school.students import RashStudent

import numpy as np

if __name__ == '__main__':
    timeToExam=10
    c = Classroom(1, BaseTeacher, RashStudent, nStudents=100, timeToExam=timeToExam)
    c.run(timeToExam=timeToExam, numberOfIteration=300, saveResults=False, visualiseResults=True, savePlot=False)
