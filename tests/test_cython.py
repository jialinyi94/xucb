import numpy as np
from ucb import UCBWrapper

# Initialize UCB with 5 arms
ucb = UCBWrapper(5)


# Define a reward function (this is just an example)
def reward_func(arm):
    return np.random.binomial(1, 0.1 + 0.1 * arm)


# Run 1000 steps
chosen_arms = ucb.multi_step(1000, reward_func)

# Print results
print("Arm values:", ucb.get_values())
print("Arm counts:", ucb.get_counts())
