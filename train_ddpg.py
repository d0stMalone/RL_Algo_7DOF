import os
import numpy as np
import pandas as pd
from envs.kuka_reach_env import KukaReachEnv
from rl.ddpg import DDPGAgent
import torch

# Hyperparameters
total_timesteps = 1_000_000
log_path = "ddpg_logs"
os.makedirs(log_path, exist_ok=True)

# Initialize environment
env = KukaReachEnv(render_mode=None)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
act_limit = env.action_space.high[0]

# Initialize DDPG agent
agent = DDPGAgent(obs_dim, act_dim, act_limit)

obs, _ = env.reset()
episode_rewards = []
episode_reward = 0
global_step = 0

print("🚀 Starting DDPG training...")
while global_step < total_timesteps:
    action = agent.select_action(obs)
    next_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    agent.store(obs, action, reward, next_obs, done)
    agent.update()

    obs = next_obs
    episode_reward += reward
    global_step += 1

    if done:
        episode_rewards.append(episode_reward)
        obs, _ = env.reset()
        episode_reward = 0

    if global_step % 5000 == 0:
        avg_rew = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
        print(f"Step: {global_step} | AvgReward (last 10 episodes): {avg_rew:.2f}")

# Save model
torch.save(agent.actor.state_dict(), os.path.join(log_path, "ddpg_actor.pth"))
torch.save(agent.critic.state_dict(), os.path.join(log_path, "ddpg_critic.pth"))
print("✅ Model saved to ddpg_logs/")

# Save reward log
pd.DataFrame({
    "Episode": np.arange(1, len(episode_rewards)+1),
    "Reward": episode_rewards
}).to_csv(os.path.join(log_path, "train_rewards.csv"), index=False)
print("📄 Training rewards saved to ddpg_logs/train_rewards.csv")

env.close()
