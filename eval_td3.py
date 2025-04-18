import numpy as np
import pandas as pd
import torch
import os
from envs.kuka_reach_env import KukaReachEnv
from rl.td3 import TD3Agent

# Setup
env = KukaReachEnv(render_mode="human")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
act_limit = env.action_space.high[0]

# Load TD3 model
agent = TD3Agent(obs_dim, act_dim, act_limit)
agent.actor.load_state_dict(torch.load("td3_logs/td3_actor.pth"))
agent.actor.eval()

# Evaluate
n_episodes = 20
results = []
print("🎯 Evaluating TD3...")

for ep in range(n_episodes):
    obs, _ = env.reset()
    done = False
    ep_reward = 0
    steps = 0
    
    while not done:
        action = agent.select_action(obs)  # <-- fixed
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_reward += reward
        steps += 1


    final_dist = info.get("distance", -1)
    success = final_dist < 0.03
    results.append((ep+1, ep_reward, final_dist, success))
    print(f"Ep {ep+1}: Reward={ep_reward:.2f}, Dist={final_dist:.3f}, Success={success}")

# Save results
df = pd.DataFrame(results, columns=["Episode", "Reward", "Distance", "Success"])
df.to_csv("td3_logs/eval_summary.csv", index=False)
print("📄 Saved: td3_logs/eval_summary.csv")
env.close()
