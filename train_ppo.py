import os
import numpy as np
import pandas as pd
import torch
from rl.ppo import PPOAgent
from envs.kuka_reach_env import KukaReachEnv

# Hyperparameters
total_timesteps = 1_000_000
rollout_len = 2048
eval_freq = 5000
log_path = "ppo_logs"
os.makedirs(log_path, exist_ok=True)

# Initialize environment
env = KukaReachEnv(render_mode=None)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

# PPO Agent
agent = PPOAgent(obs_dim, act_dim)

# Training state
obs, _ = env.reset()
episode_rewards = []
episode_reward = 0
global_step = 0

print("🚀 Starting PPO training...")
while global_step < total_timesteps:
    for _ in range(rollout_len):
        action, log_prob, value = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.store(obs, action, log_prob, reward, done, value)
        episode_reward += reward
        obs = next_obs
        global_step += 1

        if done:
            episode_rewards.append(episode_reward)
            obs, _ = env.reset()
            episode_reward = 0

        if global_step >= total_timesteps:
            break

    # PPO update
    _, _, last_val = agent.select_action(obs)
    agent.update(last_val)

    if global_step % eval_freq < rollout_len:
        avg_rew = np.mean(episode_rewards[-10:]) if episode_rewards else 0.0
        print(f"🔄 Step: {global_step} | AvgReward (last 10 episodes): {avg_rew:.2f}")

# Save model weights
torch.save(agent.net.state_dict(), os.path.join(log_path, "ppo_policy.pth"))
print("✅ Model saved: ppo_logs/ppo_policy.pth")

# Save reward logs to CSV
reward_log_path = os.path.join(log_path, "train_rewards.csv")
pd.DataFrame({
    "Episode": np.arange(1, len(episode_rewards)+1),
    "Reward": episode_rewards
}).to_csv(reward_log_path, index=False)
print(f"📄 Training rewards saved to: {reward_log_path}")

env.close()
