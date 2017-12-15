import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):
    def __init__(self, dim_inputs, dim_outputs):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(dim_inputs, 64)
        self.affine2 = nn.Linear(64, 64)

        self.action_mean = nn.Linear(64, dim_outputs)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.zeros(1, dim_outputs))

        self.saved_actions = []
        self.rewards = []
        self.final_value = 0
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = self.act(self.affine1(x))
        x = self.act(self.affine2(x))

        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std


class Value(nn.Module):
    def __init__(self, dim_inputs):
        super(Value, self).__init__()
        self.affine1 = nn.Linear(dim_inputs, 64)
        self.affine2 = nn.Linear(64, 64)
        self.value_head = nn.Linear(64, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = self.act(self.affine1(x))
        x = self.act(self.affine2(x))

        state_values = self.value_head(x)
        return state_values


class Adv(nn.Module):
    def __init__(self, dim_inputs, dropout):
        super(Adv, self).__init__()
        self.affine1 = nn.Linear(dim_inputs, 32)
        self.affine2 = nn.Linear(32, 32)
        self.adv_head = nn.Linear(32, 1)

        self.act = nn.LeakyReLU()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.drop(self.act(self.affine1(x)))
        x = self.drop(self.act(self.affine2(x)))
        advantage = self.adv_head(x)

        return advantage