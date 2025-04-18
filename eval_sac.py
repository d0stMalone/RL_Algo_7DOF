import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from gymnasium.envs.registration import register
import gymnasium as gym

register(
    id="KukaReach6DOF-v0",
    entry_point="envs.kuka_reach_env:KukaReachEnv"
)

env = DummyVecEnv([lambda: Monitor(gym.make("KukaReach6DOF-v0", render_mode="human"))])

env.training = False
env.norm_reward = False

model = SAC.load("sac_ur5_3d_model", env=env)

episodes = 20
successes = 0
episode_rewards = []

for ep in range(episodes):
    obs = env.reset()
    done = False
    total_reward = 0
    final_dist = 1.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
        final_dist = info[0]["distance"]
        done = done[0]

    success = final_dist < 0.02
    successes += int(success)
    episode_rewards.append(total_reward)
    print(f"Episode {ep+1}: Reward = {total_reward:.2f} | Final Dist = {final_dist:.3f} | Success = {success}")

print(f"\n✅ Average Reward: {np.mean(episode_rewards):.2f}")
print(f"🎯 Success Rate: {successes}/{episodes} ({(successes/episodes)*100:.2f}%)")
