import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from envs.kuka_reach_env import KukaReachEnv
from rl.ppo import PPOAgent

# Load environment
env = KukaReachEnv(render_mode="human")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

# Load custom PPO agent
agent = PPOAgent(obs_dim, act_dim)
agent.net.load_state_dict(torch.load("ppo_logs/ppo_policy.pth"))

# Evaluation loop
n_episodes = 20
results = []

print("🎯 Evaluating PPO...")
for ep in range(n_episodes):
    obs, _ = env.reset()
    done = False
    ep_reward = 0
    steps = 0

    while not done:
        action, _, _ = agent.select_action(obs)  # if you support deterministic=True, add it here
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_reward += reward
        steps += 1

    final_dist = info.get("distance", -1)
    success = final_dist < 0.03
    results.append((ep + 1, ep_reward, final_dist, success))
    print(f"Ep {ep+1}: Reward={ep_reward:.2f}, Dist={final_dist:.3f}, Success={success}")

# Save results
df = pd.DataFrame(results, columns=["Episode", "Reward", "Distance", "Success"])
df.to_csv("ppo_logs/eval_summary.csv", index=False)
print("📄 Saved: ppo_logs/eval_summary.csv")
env.close()
