"""
main file
"""
from school import Classroom
from school.teachers import Teacher
from school.students import Student

if __name__ == '__main__':
    c = Classroom(7, Teacher, Student)
    # c.run(10)
    print(c)
