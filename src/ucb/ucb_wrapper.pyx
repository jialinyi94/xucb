# distutils: language = c++
# cython: language_level=3

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector

cdef extern from "ucb.cpp":
    cdef cppclass UCB:
        UCB(int n_arms)
        int select_arm()
        void update(int chosen_arm, double reward)
        vector[int] multi_step(int num_steps, const double* rewards)
        vector[double] get_values()
        vector[int] get_counts()

cdef class PyUCB:
    cdef UCB* c_ucb

    def __cinit__(self, int n_arms):
        if n_arms <= 0:
            raise ValueError("Number of arms must be positive")
        self.c_ucb = new UCB(n_arms)

    def __dealloc__(self):
        del self.c_ucb

    def select_arm(self):
        return self.c_ucb.select_arm()

    def update(self, int chosen_arm, double reward):
        if chosen_arm < 0 or chosen_arm >= len(self.get_values()):
            raise IndexError("Invalid arm index")
        self.c_ucb.update(chosen_arm, reward)

    def multi_step(self, int num_steps, np.ndarray[double, ndim=1] rewards):
        if num_steps <= 0:
            raise ValueError("Number of steps must be positive")
        if rewards.shape[0] != len(self.get_values()):
            raise ValueError("Rewards array must have the same length as the number of arms")
        return self.c_ucb.multi_step(num_steps, &rewards[0])

    def get_values(self):
        return self.c_ucb.get_values()

    def get_counts(self):
        return self.c_ucb.get_counts()
