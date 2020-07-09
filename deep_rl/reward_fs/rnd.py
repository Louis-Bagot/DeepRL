import numpy as np
import torch
from .core import Intrinsic_Reward_Function
from ..utils import *

class RND(Intrinsic_Reward_Function):
    """ Implements the RND intrinsic reward algorithm.
    RND uses as reward the prediction error a predictor makes in trying to fit
    the output of a randomly fixed target network, taking as input just the
    current state. """

    def __init__(self, config):
        self.config = config
        ## RND predictor and target nets
        self.pred_net_rnd = config.pred_net_rnd_fn()
        self.targ_net_rnd = config.targ_net_rnd_fn()

        # RND NORMALIZATION AND HYPERPARAMETERS
        if config.rnd_normalize_s:
            self.rnd_state_normalizer = MeanStdNormalizer(clip=5)

        if config.rnd_normalize_r:
            self.rnd_reward_normalizer = MeanStdNormalizer(clip=None)

        self.n_calls = 0 # number of calls to the intrinic reward function

    def learnable_params(self):
        return list(self.pred_net_rnd.parameters())

    def __call__(self, state, action, next_state):
        """
        Computes the RND intrinsic reward based on the s,a,s' transition.
        In order to normalize the states and rewards, we use running mean and
        std estimates.
        """
        self.n_calls += 1
        state = self.config.state_normalizer((state,))
        if self.config.rnd_normalize_s:
            state = self.rnd_state_normalizer(state)

        with torch.no_grad():
            pred_state = self.pred_net_rnd(state)
            targ_state = self.targ_net_rnd(state)
            reward = to_np((pred_state - targ_state).pow(2).sum())
            if self.config.rnd_normalize_r:
                reward = self.rnd_reward_normalizer([[reward]])[0,0]

        if self.n_calls <= self.config.steps_before_rnd:
            return 0 # ignore intrinsic reward when we do not have enough data

        return reward

    def compute_loss(self, states, actions, next_states):
        config = self.config # shorthand
        ## Compute the RND loss
        pred_rnd = self.pred_net_rnd(states)
        with torch.no_grad():
            targ_rnd = self.targ_net_rnd(states)
        loss_rnd = (pred_rnd - targ_rnd).pow(2)
        ### Mask out results (don't learn too fast /o/)
        mask = torch.rand(loss_rnd.shape[1]).to(config.DEVICE) # keep only every 4th state
        mask = (mask < config.rnd_update_p).float()
        loss_rnd = (loss_rnd * mask).sum() / mask.sum().clamp(min=1)
        return loss_rnd
