from . import Classroom
from .students import Student
from .teachers import Teacher

'''
This file is only for testing
running by: python -m school
'''

c = Classroom(7, Teacher, Student)
print(c)
