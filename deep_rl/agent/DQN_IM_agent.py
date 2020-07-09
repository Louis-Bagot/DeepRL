#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from ..utils import *
import time
from .BaseAgent import *
from .DQN_agent import *


class DQN_IM_Actor(BaseActor):
    """ No changes from the DQNActor here """
    def __init__(self, config):
        BaseActor.__init__(self, config)
        self.config = config
        self.start()

    def _transition(self):
        if self._state is None:
            self._state = self._task.reset()
        config = self.config
        with config.lock:
            q_values = self._network(config.state_normalizer(self._state))
        q_values = to_np(q_values).flatten()
        if self._total_steps < config.exploration_steps \
                or np.random.rand() < config.random_action_prob():
            action = np.random.randint(0, len(q_values))
        else:
            action = np.argmax(q_values)
        next_state, reward, done, info = self._task.step([action])
        entry = [self._state[0], action, reward[0], next_state[0], int(done[0]), info]
        self._total_steps += 1
        self._state = next_state
        return entry


class DQN_IM_Agent(BaseAgent):
    """ DQN with intrinsic motivation.
    The intrinsic motivation function fir generates an intrinsic reward
    fir(s,a,s') = R_i from each transition.
    Writing R_e the extrinsic reward, the total reward is then
    R = coef_r * R_e + coef_r_i * R_i """
    def __init__(self, config):
        # CONFIG STUFF
        BaseAgent.__init__(self, config)
        self.config = config
        config.lock = mp.Lock()

        # REPLAY
        self.replay = config.replay_fn()

        # NETWORKS & OPTIMIZER INITIALIZATIONS
        ## DQN net and target nets
        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        ## Intrinsic reward function inits; weight in init
        self.coef_r = config.coef_r # reward coefficient in the total reward r = w1*r+w2*r_i
        self.coef_r_i = config.coef_r_i # intrinsic reward coef (beta)
        self.fir = config.fir(config) # intrinsic reward function

        ## Optimizer with all learnable parameters
        total_params = list(self.network.parameters()) + self.fir.learnable_params()
        self.optimizer = config.optimizer_fn(total_params)

        # ACTOR
        self.actor = DQN_IM_Actor(config)
        self.actor.set_network(self.network)

        # OTHER PARAMETER INITS
        self.total_steps = 0
        self.batch_indices = range_tensor(self.replay.batch_size)

    def close(self):
        close_obj(self.replay)
        close_obj(self.actor)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        q = self.network(state)
        action = to_np(q.argmax(-1))
        self.config.state_normalizer.unset_read_only()
        return action

    def step(self):
        config = self.config

        # SAMPLE - Call Actor to store experiences (transitions) in the Buffer
        transitions = self.actor.step()
        experiences = []
        for state, action, reward, next_state, done, info in transitions:
            self.record_online_return(info)
            self.total_steps += 1
            reward = config.reward_normalizer(reward)
            reward_i = self.fir(state, action, next_state) # intrinsic reward
            experiences.append([state, action, reward, reward_i, next_state, done])
        self.replay.feed_batch(experiences)

        # LEARN - perform SGD
        if self.total_steps > self.config.exploration_steps:
            ## Extract batch of transitions
            experiences = self.replay.sample()
            states, actions, rewards, rewards_i, next_states, terminals = experiences

            states = self.config.state_normalizer(states)
            actions = tensor(actions).long()
            next_states = self.config.state_normalizer(next_states)
            rewards = tensor(rewards)
            rewards_i = tensor(rewards_i) # intrinsic rewards
            terminals = tensor(terminals)

            ## Compute the DQN loss
            q_next = self.target_network(next_states).detach()
            if self.config.double_q:
                best_actions = torch.argmax(self.network(next_states), dim=-1)
                q_next = q_next[self.batch_indices, best_actions]
            else:
                q_next = q_next.max(1)[0]
            total_rewards = self.coef_r*rewards + self.coef_r_i*rewards_i
            q_next = self.config.discount * q_next * (1 - terminals)
            q_next.add_(total_rewards)
            q = self.network(states)
            q = q[self.batch_indices, actions]
            loss_dqn = (q_next - q).pow(2).mul(0.5).mean()

            ## Compute fir loss and total loss
            loss_ri = self.fir.compute_loss(states, actions, next_states)
            loss = loss_dqn + loss_ri

            ## Optimize & stuff
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
            with config.lock:
                self.optimizer.step()

        if self.total_steps / self.config.sgd_update_frequency % \
                self.config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())
