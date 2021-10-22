"""
main file
"""
from school import Classroom
from school.teachers import RandomTeacher
from school.students import RashStudent

if __name__ == '__main__':
    c = Classroom(7, RandomTeacher, RashStudent, nStudents=100)
    c.run(timeToExam=100, numberOfIteration=10)
    print(c)
