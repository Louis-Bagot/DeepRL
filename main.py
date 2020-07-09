#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from deep_rl import *


# DQN
def dqn_pixel(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.async_actor = True
    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()

    #config.optimizer_fn = lambda params: torch.optim.RMSprop(
       # params, lr=0.00025, alpha=0.95, eps=0.01, centered=True)
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.00025, eps=0.01 / 32)

    # config.network_fn = lambda: VanillaNet(config.action_dim, NatureConvBody(in_channels=config.history_length))
    config.network_fn = lambda: DuelingNet(config.action_dim, NatureConvBody(in_channels=config.history_length))
    config.random_action_prob = LinearSchedule(1.0, 0.01, 1e6)

    if config.async_actor:
        config.replay_fn = lambda: AsyncReplay(memory_size=int(1e6), batch_size=32)
    else:
        config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=32)

    config.batch_size = 32
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.exploration_steps = 50000
    config.sgd_update_frequency = 4
    config.gradient_clip = 5
    config.history_length = 4
    config.double_q = True
    # config.double_q = False
    config.max_steps = int(2e7)
    run_steps(DQNAgent(config))


# DQN+RND
def dqnrnd_pixel(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.async_actor = False # !!
    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()

    #config.optimizer_fn = lambda params: torch.optim.RMSprop(
       # params, lr=0.00025, alpha=0.95, eps=0.01, centered=True)
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.00025, eps=0.01 / 32)

    # config.network_fn = lambda: VanillaNet(config.action_dim, NatureConvBody(in_channels=config.history_length))
    config.network_fn = lambda: DuelingNet(config.action_dim, NatureConvBody(in_channels=config.history_length))
    config.random_action_prob = LinearSchedule(1.0, 0.01, 1e6)

    if config.async_actor:
        config.replay_fn = lambda: AsyncReplay(memory_size=int(1e6), batch_size=32)
    else:
        config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=32) # !!

    config.batch_size = 32
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.exploration_steps = 50000
    config.sgd_update_frequency = 4
    config.gradient_clip = 5
    config.history_length = 4
    config.double_q = True
    # config.double_q = False
    config.max_steps = int(2e7)
    # RND stuff
    config.pred_net_rnd_fn = lambda: VanillaNet(config.action_dim, NatureConvBody(in_channels=config.history_length))
    config.targ_net_rnd_fn = lambda: VanillaNet(config.action_dim, JustConvBody(in_channels=config.history_length))
    config.coef_r = 2. # !!
    config.coef_rrnd = 1. # !!

    run_steps(DQNRNDAgent(config))


# DQN+IM
def dqn_im_pixel(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.async_actor = False # !!
    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()

    #config.optimizer_fn = lambda params: torch.optim.RMSprop(
       # params, lr=0.00025, alpha=0.95, eps=0.01, centered=True)
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.00025, eps=0.01 / 32)

    # config.network_fn = lambda: VanillaNet(config.action_dim, NatureConvBody(in_channels=config.history_length))
    config.network_fn = lambda: DuelingNet(config.action_dim, NatureConvBody(in_channels=config.history_length))
    config.random_action_prob = LinearSchedule(1.0, 0.01, 1e6)

    if config.async_actor:
        config.replay_fn = lambda: AsyncReplay(memory_size=int(1e6), batch_size=32)
    else:
        config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=32) # !!

    config.batch_size = 32
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.exploration_steps = 50000
    config.sgd_update_frequency = 4
    config.gradient_clip = 5
    config.history_length = 4
    config.double_q = True
    # config.double_q = False
    config.max_steps = int(2e7)
    # RND stuff
    config.pred_net_rnd_fn = lambda: VanillaNet(config.action_dim, NatureConvBody(in_channels=config.history_length))
    config.targ_net_rnd_fn = lambda: VanillaNet(config.action_dim, JustConvBody(in_channels=config.history_length))
    config.fir = RND
    config.coef_r = 2. # !!
    config.coef_ri = 1. # !!

    run_steps(DQN_IM_Agent(config))


if __name__ == '__main__':
    local = False
    LOGDIR = './' if local else '/project/'


    mkdir(LOGDIR+'log')
    mkdir(LOGDIR+'tf_log')

    set_one_thread()
    random_seed()
    # select_device(-1)
    select_device(0)

    game = 'BreakoutNoFrameskip-v4'
    # dqn_pixel(game=game)
    # dqnrnd_pixel(game=game)
    dqn_im_pixel(game=game)
