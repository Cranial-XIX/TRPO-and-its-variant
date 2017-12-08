import gym

from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from utils.evaluation_utils import evaluate_policy
from env_wrappers.swimmer_wrapper import SwimmerEnv
from models.models import FeedForwardRegressor, FeedForwardSoftmax
from trpo_agent import TRPOAgent

def main():
  action_space_n = 625
  observation_dim = 8
  env = SwimmerEnv(action_space_n)

  policy_model = FeedForwardSoftmax(observation_dim, action_space_n)
  value_function_model = FeedForwardRegressor(observation_dim)
  agent = TRPOAgent(env, policy_model, value_function_model)

  while(True):
    policy, diagnostics = agent.step()
    r = evaluate_policy(env, policy, 1000, 0.95, 1)
    print("Evaluation avg reward = %f "% r)
    for key, value in diagnostics.iteritems():
      print("{}: {}".format(key, value))

if __name__ == "__main__":
  main()
