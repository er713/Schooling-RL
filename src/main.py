"""
main file
"""
from school import Classroom, Plotter, import_results
from school.teachers import *
from school.students import RashStudent

import numpy as np

if __name__ == '__main__':
   
    c = Classroom(1, RandomTeacher, RashStudent, nStudents=100)
    c.run(timeToExam=6, numberOfIteration=100, saveResults=False, visualiseResults=True, savePlot=False)
