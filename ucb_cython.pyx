# File: ucb_cython.pyx

import numpy as np
cimport numpy as np
cimport cython

# Explicitly declare the data types for NumPy arrays
ctypedef np.float64_t DTYPE_float64_t
ctypedef np.int64_t DTYPE_int64_t

@cython.boundscheck(False)
@cython.wraparound(False)
def run_ucb(int n_arms, int n_rounds, np.ndarray[DTYPE_float64_t, ndim=2] rewards_matrix):
    cdef np.ndarray[DTYPE_float64_t, ndim=1] rewards = np.zeros(n_arms, dtype=np.float64)
    cdef np.ndarray[DTYPE_int64_t, ndim=1] counts = np.zeros(n_arms, dtype=np.int64)
    cdef DTYPE_int64_t total_count = 0
    cdef int chosen_arm
    cdef DTYPE_float64_t reward
    cdef np.ndarray[DTYPE_float64_t, ndim=1] ucb_values = np.zeros(n_arms, dtype=np.float64)
    cdef np.ndarray[DTYPE_int64_t, ndim=1] chosen_arms = np.zeros(n_rounds, dtype=np.int64)
    
    cdef int i
    for i in range(n_rounds):
        if total_count < n_arms:
            chosen_arm = total_count
        else:
            nonzero_counts = counts > 0
            ucb_values[nonzero_counts] = (rewards[nonzero_counts] / counts[nonzero_counts] + 
                                          np.sqrt(2 * np.log(total_count) / counts[nonzero_counts]))
            ucb_values[~nonzero_counts] = np.inf
            chosen_arm = np.argmax(ucb_values)
        
        reward = rewards_matrix[i, chosen_arm]
        rewards[chosen_arm] += reward
        counts[chosen_arm] += 1
        total_count += 1
        chosen_arms[i] = chosen_arm
    
    return chosen_arms, rewards, counts

# Python wrapper class
class UCB:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.rewards = None
        self.counts = None
    
    def run(self, n_rounds, rewards_matrix):
        rewards_matrix = np.asarray(rewards_matrix, dtype=np.float64)  # Ensure float64 type
        chosen_arms, self.rewards, self.counts = run_ucb(self.n_arms, n_rounds, rewards_matrix)
        return chosen_arms

    def get_estimated_probs(self):
        return self.rewards / self.counts if self.counts is not None else None