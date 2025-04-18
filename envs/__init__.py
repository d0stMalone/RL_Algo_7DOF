from envs.kuka_reach_env import KukaReachEnv
import gym
from gym.envs.registration import register

register(
    id='KukaReach-v0',
    entry_point='envs.kuka_reach_env:KukaReachEnv',
)
