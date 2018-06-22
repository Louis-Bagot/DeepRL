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

class BootstrappedQRDQN(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.target_network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()
        self.policy = config.policy_fn()
        self.total_steps = 0
        self.quantile_weight = 1.0 / self.config.num_quantiles
        self.cumulative_density = self.network.tensor(
            (2 * np.arange(self.config.num_quantiles) + 1) / (2.0 * self.config.num_quantiles))

        self.option = self.network.tensor(np.random.randint(config.num_options)).long()

    def option_to_q_values(self, options, quantiles):
        config = self.config
        if config.num_quantiles % config.num_options:
            raise Exception('Smoothed quantile options is not supported')
        quantiles = quantiles.view(quantiles.size(0), quantiles.size(1), config.num_options, -1)
        quantiles = quantiles.mean(-1)
        q_values = quantiles[self.network.range(quantiles.size(0)), :, options]
        return q_values

    def huber(self, x):
        cond = (x.abs() < 1.0).float().detach()
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1 - cond)

    def evaluation_action(self, state):
        quantile_values = self.network.predict(np.stack([self.config.state_normalizer(state)])).squeeze(0).detach()
        q_values = quantile_values.sum(-1).cpu().detach().numpy()
        return np.argmax(q_values.flatten())

    def episode(self, deterministic=False):
        episode_start_time = time.time()
        config = self.config
        state = self.task.reset()
        total_reward = 0.0
        steps = 0
        while True:
            quantile_values, option_values = self.network.predict(np.stack([self.config.state_normalizer(state)]))

            greedy_option = torch.argmax(option_values, dim=-1)
            random_option_prob = config.random_option_prob()
            random_option = self.network.tensor(np.random.randint(
                config.num_options, size=config.num_workers)).long()
            dice = self.network.tensor(np.random.rand(config.num_workers))
            if dice < random_option_prob:
                new_option = random_option
            else:
                new_option = greedy_option
            if config.option_type == 'constant_beta':
                if steps == 0 or np.random.rand() < config.beta:
                    self.option = new_option

            if self.total_steps < self.config.exploration_steps:
                action = np.random.randint(0, self.task.action_dim)
            else:
                if config.option_type is not None:
                    q_values = self.option_to_q_values(self.option, quantile_values)
                else:
                    q_values = (quantile_values * self.quantile_weight).sum(-1)
                q_values = q_values.cpu().detach().numpy()
                action = self.policy.sample(q_values.flatten())
            next_state, reward, done, _ = self.task.step(action)
            total_reward += reward
            reward = self.config.reward_normalizer(reward)
            if not deterministic:
                self.replay.feed([state, action, reward, next_state, int(done), np.asscalar(
                    self.option.detach().cpu().numpy())])
                self.total_steps += 1
            steps += 1
            state = next_state

            if not deterministic and self.total_steps > self.config.exploration_steps and \
                    self.total_steps % config.sgd_update_frequency == 0:
                experiences = self.replay.sample()
                states, actions, rewards, next_states, terminals, options = experiences
                states = self.config.state_normalizer(states)
                next_states = self.config.state_normalizer(next_states)

                quantiles_next, option_values_next = self.target_network.predict(next_states)
                quantiles_next = quantiles_next.detach()
                a_next = torch.argmax(quantiles_next.sum(-1), dim=-1)
                quantiles_next = quantiles_next[self.network.range(quantiles_next.size(0)), a_next, :]

                rewards = self.network.tensor(rewards)
                terminals = self.network.tensor(terminals)
                quantiles_next = rewards.view(-1, 1) + self.config.discount * (1 - terminals.view(-1, 1)) * quantiles_next

                quantiles, option_values = self.network.predict(states)
                actions = self.network.tensor(actions).long()
                quantiles = quantiles[self.network.range(quantiles.size(0)), actions, :]

                quantiles_next = quantiles_next.t().unsqueeze(-1)
                diff = quantiles_next - quantiles
                loss = self.huber(diff) * (self.cumulative_density.view(1, -1) - (diff.detach() < 0).float()).abs()
                loss = loss.mean(0).mean(0).sum()

                if config.option_type is not None:
                    options = self.network.tensor(options).long()
                    option_values_next = config.beta * option_values_next.max(1)[0] + (1 - config.beta) * \
                                         option_values_next[self.network.range(options.size(0)), options]
                    option_values_next = option_values_next.unsqueeze(-1).detach()
                    option_values_next = rewards.view(-1, 1) + self.config.discount * (1 - terminals.view(-1, 1)) * option_values_next
                    option_values = option_values[self.network.range(options.size(0)), options].unsqueeze(1)
                    option_loss = (option_values - option_values_next).pow(2).mul(0.5).mean()
                    loss = loss + option_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.evaluate()
            if not deterministic and self.total_steps % self.config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())
            if not deterministic:
                self.policy.update_epsilon()

            if done:
                break

        episode_time = time.time() - episode_start_time
        self.config.logger.debug('episode steps %d, episode time %f, time per step %f' %
                          (steps, episode_time, episode_time / float(steps)))
        return total_reward, steps