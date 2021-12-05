from typing import *
from collections import namedtuple, deque
import random

class Record:
    def __init__(self, state: List[float] = None, action: int = None):
        self.state = state
        self.action = action


class ExamResultsRecord(Record):

    def __init__(self, state: List[float] = None, action: int = None):
        super(ExamResultsRecord, self).__init__(state, action)
        self.noAcquiredExamResults = 0
        self.marksSum = 0

    def clear(self):
        self.state = None
        self.action = None
        self.noAcquiredExamResults = 0
        self.marksSum = 0


class MemoryRecord(Record):

    def __init__(self, state: List[float] = None, action: int = None, reward: int = None,
                 nState: List[float] = None, done: bool = None):
        super(MemoryRecord, self).__init__(state, action)
        self.reward = reward
        self.nextState = nState
        self.done = done

class ReplayBuffer:

    def __init__(self, action_size, buffer_size, batch_size):
       
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = [e.state for e in experiences if e is not None]
        actions = [e.action for e in experiences if e is not None]
        rewards = [e.reward for e in experiences if e is not None]
        next_states = [e.next_state for e in experiences if e is not None]
        dones = [e.done for e in experiences if e is not None]
    
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)