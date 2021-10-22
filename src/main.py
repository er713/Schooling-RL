"""
main file
"""
from school import Classroom
from school.teachers import RandomTeacher
from school.students import RashStudent

if __name__ == '__main__':
    c = Classroom(7, RandomTeacher, RashStudent)
    c.run(10)
    print(c)
