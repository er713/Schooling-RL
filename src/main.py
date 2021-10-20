"""
main file
"""
from school import Classroom
from school.teachers import RandomTeacher
from school.students import Student

if __name__ == '__main__':
    c = Classroom(7, RandomTeacher, Student)
    # c.run(10)
    print(c)
