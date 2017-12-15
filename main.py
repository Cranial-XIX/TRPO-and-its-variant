import argparse
import csv
import gym
import itertools
import numpy as np
import scipy.optimize
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from trpo import trpo_step
from utils import *

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True
torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='TRPO and its variants')

parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                    help='discount factor (default: 0.995)')

parser.add_argument('--env-name', default="Swimmer-v1", metavar='G',
                    help='the Mujuco environment to run (default: Swimmer-v1)')

parser.add_argument('--algo', default="trpo-kl", metavar='G',
                    help='which algorithm to use {trpo-kl, trpo-mse, pure-opt, adv-estimate}')

parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                    help='gae (default: 0.97)')

parser.add_argument('--dropout', type=float, default=0.3, metavar='G',
                    help='advantage neural net dropout rate (default: 0.3)')

parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')

parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                    help='max kl value (default: 1e-2)')

parser.add_argument('--damping', type=float, default=1e-1, metavar='G',
                    help='damping (default: 1e-1)')

parser.add_argument('--seed', type=int, default=1904, metavar='N',
                    help='random seed (default: 1904)')

parser.add_argument('--batch-size', type=int, default=15000, metavar='N',
                    help='batch size (deafult: 15000)')

parser.add_argument('--render', action='store_true',
                    help='render the environment')

parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 1)')

args = parser.parse_args()

env = gym.make(args.env_name)

dim_observations = env.observation_space.shape[0]
dim_actions = env.action_space.shape[0]

env.seed(args.seed)
torch.manual_seed(args.seed)

policy_net = Policy(dim_observations, dim_actions)
value_net = Value(dim_observations)

advantage_net = Adv(dim_observations + dim_actions, args.dropout)

# get model paramters
a_params = itertools.ifilter(
    lambda x: x.requires_grad, advantage_net.parameters()
)

adv_l2 = 0.02
# define the optimizer to use; currently use Adam
opt_a = optim.Adam(
    a_params, lr=1e-4, weight_decay=adv_l2
)

mse = nn.MSELoss()

def sample_action(state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action


def adv_estimate_linesearch(model,
               f,
               x,
               fullstep,
               max_backtracks=10,
               accept_ratio=.1):
    fval = f(True).data
    #print("fval before", fval[0])
    max_improve = -1000
    max_x = x
    for (_n_backtracks, stepfrac) in enumerate((0.5**np.arange(max_backtracks))):
        xnew = x + stepfrac * fullstep
        set_flat_params_to(model, xnew)
        newfval = f(True).data
        actual_improve = fval - newfval
        if actual_improve[0]> max_improve:
            max_x = xnew
            max_improve = actual_improve[0]

    if max_improve > -1000:
        print "max_improve: ", max_improve
        return True, max_x
    else:
        return False, x

def update_params(batch):
    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(batch.state)
    values = value_net(Variable(states))

    returns = torch.Tensor(actions.size(0),1)
    deltas = torch.Tensor(actions.size(0),1)
    advantages = torch.Tensor(actions.size(0),1)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]

        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    adv_loss_data = 0
    if args.algo == 'adv-estimate':
        # optimize advantage net
        advantage_net.train()
        opt_a.zero_grad()

        target_advs = Variable(advantages, requires_grad=False)

        our_advs = advantage_net(
            torch.cat(
                (Variable(states), Variable(actions)), 1
            )
        )

        adv_loss = F.smooth_l1_loss(our_advs, target_advs)
        adv_loss_data = adv_loss.data[0]
        print "advantage huber loss : ", adv_loss_data
        adv_loss.backward()
        opt_a.step()
        advantage_net.eval()

    # optimize value net, original code uses the same LBFGS to optimize the value loss
    targets = Variable(returns)
    def get_value_loss(flat_params):
        set_flat_params_to(value_net, torch.Tensor(flat_params))
        for param in value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

        values_ = value_net(Variable(states))

        value_loss = (values_ - targets).pow(2).mean()

        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * args.l2_reg
        value_loss.backward()
        return (value_loss.data.double().numpy()[0], get_flat_grad_from(value_net).data.double().numpy())

    flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss, get_flat_params_from(value_net).double().numpy(), maxiter=25)
    set_flat_params_to(value_net, torch.Tensor(flat_params))

    copy_advantages = advantages
    advantages = (advantages - advantages.mean()) / advantages.std()

    action_means, action_log_stds, action_stds = policy_net(Variable(states))
    fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

    def get_loss(volatile=False):
        action_means, action_log_stds, action_stds = policy_net(Variable(states, volatile=volatile))
        log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
        action_loss = -Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))
        return action_loss.mean()


    def get_kl():
        mean1, log_std1, std1 = policy_net(Variable(states))

        mean0 = Variable(mean1.data)
        log_std0 = Variable(log_std1.data)
        std0 = Variable(std1.data)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)


    if not args.algo == 'adv-estimate': 
        
        trpo_step(policy_net, get_loss, get_kl, args.max_kl, args.damping, args.algo)

    elif adv_loss_data > 1:

        trpo_step(policy_net, get_loss, get_kl, args.max_kl, args.damping, 'trpo-kl')

    else:

        action_means, action_log_stds, action_stds = policy_net(Variable(states))

        estimated_advantages = advantage_net(
            torch.cat(
                (Variable(states), action_means), 1
            )
        )

        estimated_advantages = Variable(estimated_advantages.data)
        adv_var = Variable(copy_advantages)


        log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
        delta = adv_var - estimated_advantages
        mask = delta > 0.05

        loss = -torch.sum(torch.masked_select(log_prob * adv_var, mask))

        grads = torch.autograd.grad(loss, policy_net.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).data
        stepdir = -loss_grad

        mse_norm = torch.sum(loss_grad * loss_grad)
        lm = np.sqrt(mse_norm/args.max_kl)
        fullstep = stepdir / lm

        prev_params = get_flat_params_from(policy_net)
        success, new_params = adv_estimate_linesearch(policy_net, get_loss, prev_params, fullstep)
        print "success: ", success
        set_flat_params_to(policy_net, new_params)


running_state = ZFilter((dim_observations,), clip=5)
running_reward = ZFilter((1,), demean=False, clip=10)

fname = args.algo + '_' + args.env_name + '_' + str(adv_l2) + '_seed=' + str(args.seed) + '.csv'
template = 'Episode {}\tLast reward: {}\tAverage reward {:.2f}'

with open(fname, 'w') as csvfile:

    cols = ['epi', 'rew']
    writer = csv.DictWriter(csvfile, fieldnames = cols)
    writer.writeheader()


    for i_episode in xrange(201):

        memory = Memory()
        num_steps = 0
        reward_batch = 0
        num_episodes = 0

        while num_steps < args.batch_size:
            state = env.reset()
            state = running_state(state)

            reward_sum = 0
            for t in range(5000): # upper limit of steps for one episode
                action = sample_action(state)
                action = action.data[0].numpy()
                next_state, reward, done, _ = env.step(action)
                reward_sum += reward

                next_state = running_state(next_state)

                mask = 1
                if done:
                    mask = 0

                memory.push(state, np.array([action]), mask, next_state, reward)

                if args.render:
                    env.render()
                if done:
                    break

                state = next_state

            num_steps += (t-1)
            num_episodes += 1
            reward_batch += reward_sum

        reward_batch /= num_episodes
        batch = memory.sample()

        update_params(batch)
        #if reward_batch > 290:
        #    args.render = True

        if i_episode % args.log_interval == 0:
            print(template.format(i_episode, reward_sum, reward_batch))
            writer.writerow({'epi' : i_episode, 'rew' : reward_batch})