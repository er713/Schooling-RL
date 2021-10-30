"""
main file
"""
from school import Classroom
from school.teachers import RandomTeacher, BaseTeacher
from school.students import RashStudent

if __name__ == '__main__':
    c = Classroom(7, RandomTeacher, RashStudent, nStudents=100)
    c.run(timeToExam=100, numberOfIteration=10)
    # res = Classroom.static_import_result('./data/RandomTeacher/', 'RashStudent__100_7__2021-10-30_23-39.csv')
    # print(res[0].mark, res[0].isExam)
