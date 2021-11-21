from typing import *


class Record:
    def __init__(self, state: List[float] = None, action: int = None):
        self.state = state
        self.action = action


class ExamResultsRecord(Record):

    def __init__(self, state: List[float] = None, action: int = None):
        super(ExamResultsRecord, self).__init__(state, action)
        self.noAcquiredExamResults=0
        self.marksSum=0

    def clear(self):
        self.state=None
        self.action=None
        self.noAcquiredExamResults=0
        self.marksSum=0


class MemoryRecord(Record):

    def __init__(self, state: List[float] = None, action: int = None, reward: int = None,
                 nState: List[float] = None, done: bool = None):
        super(MemoryRecord, self).__init__(state, action)
        self.reward = reward
        self.nextState = nState
        self.done = done
