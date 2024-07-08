import numpy as np
from .ucb_wrapper import PyUCB


class UCBWrapper:
    def __init__(self, n_arms):
        """
        Initialize the UCB algorithm.

        Args:
            n_arms (int): Number of arms in the multi-armed bandit problem.
        """
        self.ucb = PyUCB(n_arms)

    def select_arm(self):
        """
        Select an arm according to the UCB algorithm.

        Returns:
            int: The index of the selected arm.
        """
        return self.ucb.select_arm()

    def update(self, chosen_arm, reward):
        """
        Update the UCB algorithm with the result of the last action.

        Args:
            chosen_arm (int): The index of the arm that was chosen.
            reward (float): The reward that was received.
        """
        self.ucb.update(chosen_arm, reward)

    def multi_step(self, num_steps, reward_func):
        """
        Run multiple steps of the UCB algorithm.

        Args:
            num_steps (int): Number of steps to run.
            reward_func (callable): 
            A function that takes an arm index and returns a reward.

        Returns:
            list: The indices of the arms that were chosen.
        """
        n_arms = len(self.ucb.get_values())
        rewards = np.array([reward_func(i) for i in range(n_arms)])
        return self.ucb.multi_step(num_steps, rewards)

    def get_values(self):
        """
        Get the current estimated values for each arm.

        Returns:
            list: The estimated values.
        """
        return self.ucb.get_values()

    def get_counts(self):
        """
        Get the number of times each arm has been pulled.

        Returns:
            list: The counts for each arm.
        """
        return self.ucb.get_counts()
