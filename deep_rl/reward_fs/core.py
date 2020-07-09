import numpy as np
import torch

class Intrinsic_Reward_Function():
    """ Exploration Bonus as a Reward signal."""

    def __init__(self, config):
        """ Initializes the intrinsic reward function.
        E.g. networs if learnt, hyperparameters, normalizers.
        """
        self.config = config
        # ...

    def learnable_params(self):
        """ Returns as a list the parameters that are learnable in the function.
        Defaults to none (empty list)
        """
        return list()

    def __call__(self, state, action, next_state):
        """ Main function, returns the intrinsic reward for this transition.
        Acts as the extrinsic reward function: from the s,a,s' transition,
        gives out a scalar. No learning is done here.
        """
        raise NotImplementedError("fir not implemented")

    def compute_loss(self, states, actions, next_states):
        """ Returns the loss to be backpropagated.
        The inputs are batches of transitions.
        The main Agent will call the optimizer, no need to do it here.
        If other forms of learning can be done, they should be done here.
        """
        pass
