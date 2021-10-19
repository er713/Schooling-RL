"""
main file
"""
from school import Classroom
from school.teachers import Teacher
from school.students import Student

if __name__ == '__main__':
    Classroom(7, Teacher, Student)
