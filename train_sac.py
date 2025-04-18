import os
import numpy as np
import pandas as pd
from envs.kuka_reach_env import KukaReachEnv
from rl.sac import SACAgent
import torch

# Hyperparameters
total_timesteps = 1_000_000
log_path = "sac_logs"
os.makedirs(log_path, exist_ok=True)

# Initialize environment
env = KukaReachEnv(render_mode=None)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
act_limit = env.action_space.high[0]

# Initialize SAC agent
agent = SACAgent(obs_dim, act_dim, act_limit)

obs, _ = env.reset()
episode_rewards = []
episode_reward = 0
global_step = 0

print("🚀 Starting SAC training...")
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
torch.save(agent.actor.state_dict(), os.path.join(log_path, "sac_actor.pth"))
torch.save(agent.critic1.state_dict(), os.path.join(log_path, "sac_critic1.pth"))
torch.save(agent.critic2.state_dict(), os.path.join(log_path, "sac_critic2.pth"))
print("✅ SAC model saved to sac_logs/")

# Save reward log
pd.DataFrame({
    "Episode": np.arange(1, len(episode_rewards)+1),
    "Reward": episode_rewards
}).to_csv(os.path.join(log_path, "train_rewards.csv"), index=False)
print(f"📄 Training rewards saved to sac_logs/train_rewards.csv")

env.close()
