from .teacher import Teacher
from typing import List
from ..task import Task
from .models.dqn import *
from .losses import dqn_loss
import tensorflow as tf
from random import shuffle
from .. import Result
from copy import deepcopy
from .utils.dqn_structs import *
from random import random, randint


class DQNTeacher(Teacher):
    """Deep Q-learning agent."""
    def __init__(self, nSkills: int, tasks: List[Task], batchSize: int = 100, timeToExam=-1, noExamTasks=-1, **kwargs):
        """Set parameters, initialize network."""
        super().__init__(nSkills, tasks, **kwargs)
        self.memSize = 300
        self.noTargetIte = 2000
        self.noLearnIte = 100
        self.epsilon=0.2
        self.batchSize = batchSize
        self.lossFun = dqn_loss
        modelInputSize=len(tasks)*2
        self.estimator = DQN(modelInputSize)
        self.targetEstimator = DQN(modelInputSize)
        self.studentsStates = {}
        self.studentsPrevState = {}
        self.beforeExamTasks = {}
        self.learnCounter = 0
        self.gamma = 0.9
        self.mem = [MemoryRecord() for _ in range(self.memSize)]
        self.memIdx = -1
        self.batchSelectIdxs = list(range(self.memSize))
        self.timeToExam=timeToExam
        self.noTasks= len(tasks)
        self.noExamTasks = noExamTasks
        self.memFull=False
        self.__examResults={}
        self.__statesBuff=[]*self.batchSize
        self.__actionsBuff=[]*self.batchSize
        self.__realQsBuff=[]*self.batchSize
        self.__targetCounter = 0
        self.__innerStudentsStates = {}

    def choose_task(self, student) -> Task:
        if student.id not in self.studentsStates:
            self.studentsStates[student.id]=self.__get_new_student_state()
        # exploitation
        if random() > self.epsilon:
            studentState = tf.expand_dims(tf.constant(self.studentsStates[student.id]), axis=0)
            actionsValues = self.estimator(studentState)[0]
            chosenTaskId = tf.argmax(actionsValues).numpy()
        # exploration
        else:
            chosenTaskId = randint(0, self.noTasks-1)
        return chosenTaskId

    def receive_result(self, result, last=False, reward=None) -> None:
        # Exam results need to be reduced in receive_exam_res
        if result.isExam:
            self.__receive_exam_res(result)
            return
        self.__update_memory(result, last)
        # If it was last task, exam result from following tasks
        # are needed to finish this record and save it into memory
        if last:
            return
        # copy estimator weights to target estimator after noTargetIte iterations
        self.__update_target()
        # train estimator with batch generated from memory
        if self.learnCounter == self.noLearnIte and self.memFull:
            batch_tuple=self.__get_batch_tuple()
            self.estimator.train_step(*batch_tuple)

    def __receive_exam_res(self, result:Result):
        # examRecord must exist because it was created when last task's result was received
        examResultRecord=self.__examResults[result.idStudent]
        examResultRecord.noAcquiredExamResults+=1
        examResultRecord.marksSum+=result.mark
        # if it is last exam task add record to memory
        if examResultRecord.noAcquiredExamResults == self.noExamTasks:
            self.__update_memory(result, last=True)

    def __get_batch_tuple(self):
        # selection of record for this learning batch
        shuffle(self.batchSelectIdxs)
        batchSelectIdxs = self.batchSelectIdxs[:self.batchSize]
        for batchIdx, memIdx in enumerate(batchSelectIdxs):
            memRecord=self.mem[memIdx]
            reward = memRecord.reward
            # calculate target q_value
            if memRecord.done:
                realQ=reward
            else:
                nextState = tf.expand_dims(tf.constant(memRecord.nextState), axis=0)
                realQ=reward+self.__get_target_q(nextState)
            # update lists buffers
            self.__statesBuff[batchIdx]=memRecord.state
            self.__actionsBuff[batchIdx]=(batchIdx, memRecord.action)
            self.__realQsBuff[batchIdx]=realQ
        # create tensors for graph execution
        states=tf.constant(self.__statesBuff)
        idxedActions=tf.constant(self.__actionsBuff)
        realQs=tf.constant(self.__realQsBuff)

        return states, idxedActions, realQs

    #tf.function
    def __get_target_q(self, state):
        bestActionIdx=tf.argmax(self.estimator(state))
        targetQ=self.targetEstimator(state)[bestActionIdx]
        return targetQ

    def __update_student_state(self, result):
        """
            studentState=[ nTF_0, sF_0, nTF_1, sF_1, ...]
            _x - task number
            nTF_x - noTriesFraction = noTries/maxNoTries
            sF_x - successFraction = noSuccesses/noTries
            maxNoTries=timeToExam
            :param result: task result
            :return: updated student state
        """
        noTries, noSuccesses = self.__update_inner_student_state(result)
        if result.isExam:
            self.studentsStates.pop(result.idStudent)
            studentState=None
        else:
            studentState=self.studentsStates[result.idStudent]
            noTriesFractionIdx = result.task.id*2
            successFractionIdx = noTriesFractionIdx+1
            studentState[noTriesFractionIdx]=noTries/self.timeToExam
            studentState[successFractionIdx]=noSuccesses/noTries
        return studentState

    def __update_inner_student_state(self, result)-> (int, int):
        """
                studentInnerState=[ ... numberOfTries_i, numberOfSuccesses_i ... ]
                :param result: task result
                :return: updated studentInnerState
        """
        noTries = None
        noSuccesses = None
        if result.isExam:
            self.__innerStudentsStates.pop(result.idStudent)
        else:
            if result.idStudent not in self.__innerStudentsStates:
                self.__innerStudentsStates[result.idStudent]=self.__get_new_student_state()
            innerStudentState = self.__innerStudentsStates[result.idStudent]
            noTryIdx = result.task.id * 2
            noSuccessesIdx = noTryIdx + 1
            innerStudentState[noTryIdx] += 1
            innerStudentState[noSuccessesIdx] += result.mark
            noTries=innerStudentState[noTryIdx]
            noSuccesses=innerStudentState[noSuccessesIdx]
        return noTries, noSuccesses

    def __get_new_student_state(self):
        # information about task consists of fraction of tries and fraction of successes
        return [0]*(2*self.noTasks)

    def __update_target(self):
        """
                Copy weights from online network to target networks
                after being called noTargetIte times.
        """
        self.__targetCounter += 1
        if self.__targetCounter == self.noTargetIte:
            self.__targetCounter=0
            self.targetEstimator.copy_weights(self.estimator)

    def __update_memory(self, result: Result, last: bool):
        # last exam task
        if last and result.isExam:
            examResultsRecord=self.__examResults[result.idStudent]
            state = examResultsRecord.state
            action = examResultsRecord.action
            reward = examResultsRecord.marksSum/examResultsRecord.noAcquiredExamResults
        # non exam task
        else:
            state = self.studentsStates[result.idStudent]
            action = result.task.id
            reward=0
        # last non exam task, prepare structure to receive exam results
        if last and not result.isExam:
            self.__examResults[result.idStudent] = ExamResultsRecord(state, action)
            return
        # set flag that memory has been fully filled
        if self.memIdx==self.memSize-1:
            self.memFull=True
        # increment memory idx and select updated record
        self.memIdx = (self.memIdx + 1) % self.memSize
        updatedMemRecord = self.mem[self.memIdx]
        # overwrite memory record and update student state
        updatedMemRecord.state=deepcopy(state)
        updatedMemRecord.reward = reward
        updatedMemRecord.action=deepcopy(action)
        updatedMemRecord.done = result.isExam
        # update student state with action (given task id) and result of that action
        newStudentState = self.__update_student_state(result)
        updatedMemRecord.nextState = deepcopy(newStudentState)



