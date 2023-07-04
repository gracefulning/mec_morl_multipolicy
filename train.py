#!/usr/bin/env python
# coding: utf-8



import gym, torch, numpy as np, torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import tianshou as ts
from copy import deepcopy
from tianshou.env import DummyVectorEnv
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
import os
import time
import json
import math
from tqdm import tqdm
from env import MEC_Env
from network import conv_mlp_net



edge_num = 1
expn = 'exp1'
config = 'multi-edge'
lr, epoch, batch_size = 1e-6, 1, 1024*4
train_num, test_num = 64, 1024
gamma, lr_decay = 0.9, None
buffer_size = 100000
eps_train, eps_test = 0.1, 0.00
step_per_epoch, episode_per_collect = 100*train_num*700, train_num
writer = SummaryWriter('tensor-board-log/ppo')  # tensorboard is also supported!
logger = ts.utils.BasicLogger(writer)
is_gpu = True
#ppo
gae_lambda, max_grad_norm = 0.95, 0.5
vf_coef, ent_coef = 0.5, 0.0
rew_norm, action_scaling = False, False
bound_action_method = "clip"
eps_clip, value_clip = 0.2, False
repeat_per_collect = 2
dual_clip, norm_adv = None, 0.0
recompute_adv = 0


INPUT_CH = 67
FEATURE_CH = 512
MLP_CH = 1024
class mec_net(nn.Module):
    def __init__(self, mode='actor', is_gpu=True):
        super().__init__()
        self.is_gpu = is_gpu
        self.mode = mode
        
        if self.mode == 'actor':
            self.network = conv_mlp_net(conv_in=INPUT_CH, conv_ch=FEATURE_CH, mlp_in=(edge_num+1)*FEATURE_CH,\
                                    mlp_ch=MLP_CH, out_ch=edge_num+1, block_num=3)
        else:
            self.network = conv_mlp_net(conv_in=INPUT_CH, conv_ch=FEATURE_CH, mlp_in=(edge_num+1)*FEATURE_CH,\
                                    mlp_ch=MLP_CH, out_ch=1, block_num=3)
        
    def load_model(self, filename):
        map_location=lambda storage, loc:storage
        self.load_state_dict(torch.load(filename, map_location=map_location))
        print('load model!')
    
    def save_model(self, filename):
        torch.save(self.state_dict(), filename)
        # print('save model!')

    def forward(self, obs, state=None, info={}):
        state = obs#['servers']
        state = torch.tensor(state).float()
        if self.is_gpu:
            state = state.cuda()

        logits = self.network(state)
        
        return logits, state





class Actor(nn.Module):
    def __init__(self, is_gpu=True):
        super().__init__()
        
        self.is_gpu = is_gpu

        self.net = mec_net(mode='actor')

    def load_model(self, filename):
        map_location=lambda storage, loc:storage
        self.load_state_dict(torch.load(filename, map_location=map_location))
        print('load model!')
    
    def save_model(self, filename):
        torch.save(self.state_dict(), filename)
        # print('save model!')

    def forward(self, obs, state=None, info={}):
            
        logits,_ = self.net(obs)
        logits = F.softmax(logits, dim=-1)

        return logits, state





class Critic(nn.Module):
    def __init__(self, is_gpu=True):
        super().__init__()

        self.is_gpu = is_gpu

        self.net = mec_net(mode='critic')

    def load_model(self, filename):
        map_location=lambda storage, loc:storage
        self.load_state_dict(torch.load(filename, map_location=map_location))
        print('load model!')
    
    def save_model(self, filename):
        torch.save(self.state_dict(), filename)
        # print('save model!')

    def forward(self, obs, state=None, info={}):
            
        v,_ = self.net(obs)

        return v




actor = Actor(is_gpu = is_gpu)
critic = Critic(is_gpu = is_gpu)

load_path = None

if is_gpu:
    actor.cuda()
    critic.cuda()

    
from tianshou.utils.net.common import ActorCritic
actor_critic = ActorCritic(actor, critic)

optim = torch.optim.Adam(actor_critic.parameters(), lr=lr)




dist = torch.distributions.Categorical

action_space = gym.spaces.Discrete(edge_num)

if lr_decay:
    lr_scheduler = LambdaLR(
        optim, lr_lambda=lambda epoch: lr_decay**(epoch-1)
    )
else:
    lr_scheduler = None

policy = ts.policy.PPOPolicy(actor, critic, optim, dist,
        discount_factor=gamma, max_grad_norm=max_grad_norm,
        eps_clip=eps_clip, vf_coef=vf_coef,
        ent_coef=ent_coef, reward_normalization=rew_norm,
        advantage_normalization=norm_adv, recompute_advantage=recompute_adv,
        dual_clip=dual_clip, value_clip=value_clip,
        gae_lambda=gae_lambda, action_space=action_space,
        lr_scheduler=lr_scheduler,
    )



for i in range(101):
    try:
        os.mkdir('save/pth-e%d/'%(edge_num) + expn + '/w%03d'%(i))
    except:
        pass

for wi in range(100,0-1,-2):
    
    if wi==100:
        epoch_a = epoch * 10
    else:
        epoch_a = epoch

    train_envs = DummyVectorEnv([lambda: MEC_Env(conf_name=config,w=wi/100.0,fc=4e9,fe=2e9,edge_num=edge_num) for _ in range(train_num)])
    test_envs = DummyVectorEnv([lambda: MEC_Env(conf_name=config,w=wi/100.0,fc=4e9,fe=2e9,edge_num=edge_num) for _ in range(test_num)]) 

    buffer = ts.data.VectorReplayBuffer(buffer_size, train_num)
    train_collector = ts.data.Collector(policy, train_envs, buffer)
    test_collector = ts.data.Collector(policy, test_envs)
    train_collector.collect(n_episode=train_num)

    def save_best_fn (policy):
        pass

    def test_fn(epoch, env_step):
        policy.actor.save_model('save/pth-e%d/'%(edge_num) + expn + '/w%03d/ep%02d-actor.pth'%(wi,epoch))
        policy.critic.save_model('save/pth-e%d/'%(edge_num) + expn + '/w%03d/ep%02d-critic.pth'%(wi,epoch))

    def train_fn(epoch, env_step):
        pass

    def reward_metric(rews):
        return rews

    result = ts.trainer.onpolicy_trainer(
            policy, train_collector, test_collector, epoch_a, step_per_epoch,
            repeat_per_collect, test_num, batch_size,
            episode_per_collect=episode_per_collect, save_best_fn =save_best_fn , logger=logger,
            test_fn = test_fn, test_in_train=False)

