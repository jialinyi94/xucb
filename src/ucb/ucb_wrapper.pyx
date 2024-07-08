# distutils: language = c++
# cython: language_level=3

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.stdexcept cimport invalid_argument, out_of_range

cdef extern from "ucb.cpp":
    cdef cppclass UCB:
        UCB(int n_arms) except +
        int select_arm()
        void update(int chosen_arm, double reward) except +
        vector[int] multi_step(int num_steps, const double* rewards) except +
        vector[double] get_values()
        vector[int] get_counts()

cdef class PyUCB:
    cdef UCB* c_ucb

    def __cinit__(self, int n_arms):
        self.c_ucb = new UCB(n_arms)

    def __dealloc__(self):
        del self.c_ucb

    def select_arm(self):
        return self.c_ucb.select_arm()

    def update(self, int chosen_arm, double reward):
        try:
            self.c_ucb.update(chosen_arm, reward)
        except out_of_range:
            raise IndexError("Invalid arm index")

    def multi_step(self, int num_steps, np.ndarray[double, ndim=1] rewards):
        if rewards.shape[0] != self.c_ucb.get_values().size():
            raise ValueError("Rewards array must have the same length as the number of arms")
        try:
            return self.c_ucb.multi_step(num_steps, &rewards[0])
        except invalid_argument:
            raise ValueError("Number of steps must be positive")

    def get_values(self):
        return self.c_ucb.get_values()

    def get_counts(self):
        return self.c_ucb.get_counts()