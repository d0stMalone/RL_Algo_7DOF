import argparse
import numpy as np
import pybullet as p
import gymnasium as gym
import time
import os
from gymnasium.envs.registration import register
from stable_baselines3 import SAC, PPO, TD3, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from envs.dijkstra_planner import dijkstra_plan
from envs.kuka_reach_env import KukaReachEnv

# Register environment
register(
    id="KukaReach6DOF-v0",
    entry_point="envs.kuka_reach_env:KukaReachEnv"
)

# Load any model dynamically
algo_map = {
    "sac": SAC,
    "ppo": PPO,
    "td3": TD3,
    "ddpg": DDPG
}

parser = argparse.ArgumentParser()
parser.add_argument("--algo", type=str, required=True, choices=algo_map.keys(), help="Algorithm to evaluate (sac/ppo/td3/ddpg)")
args = parser.parse_args()

algo_name = args.algo.lower()
log_dir = f"{algo_name}_kuka_logs"
model_path = f"{algo_name}_kuka_3d_model"
vecnorm_path = os.path.join(log_dir, "vecnormalize.pkl")

# Load environment and VecNormalize
dummy_env = DummyVecEnv([lambda: Monitor(gym.make("KukaReach6DOF-v0", render_mode="human", use_dijkstra=True))])
env = VecNormalize.load(vecnorm_path, dummy_env)
env.training = False
env.norm_reward = False

# Load model
model = algo_map[algo_name].load(model_path, env=env)

# Reset environment
obs = env.reset()
robot_id = env.envs[0].env.robot_id
goal_pos = env.envs[0].env.goal_pos
start_angles = env.envs[0].env.start_joint_state

# Run Dijkstra
print(f"🧭 Running Dijkstra baseline...")
dijkstra_path = dijkstra_plan(robot_id, start_angles, goal_pos)

if dijkstra_path:
    print(f"✅ Dijkstra path has {len(dijkstra_path)} waypoints")
    for joint_angles in dijkstra_path:
        for i in range(6):
            p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, joint_angles[i])
        for _ in range(10):
            p.stepSimulation()
        time.sleep(0.02)
else:
    print("❌ Dijkstra failed.")

# Run RL policy
print(f"\n🤖 Running {algo_name.upper()} policy...")
obs = env.reset()
total_reward = 0
success = False
max_steps = 200

for step in range(max_steps):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward[0]
    done = terminated[0] or truncated[0]

    if info[0]["distance"] < 0.02:
        success = True
        break

    p.stepSimulation()
    time.sleep(1.0 / 60.0)

print(f"\n✅ {algo_name.upper()} run finished:")
print(f"   🎯 Final Distance: {info[0]['distance']:.3f} m")
print(f"   💰 Total Reward: {total_reward:.2f}")
print(f"   🏁 Success: {'Yes' if success else 'No'}")

# Hold the window open
print("\n👀 Holding PyBullet GUI open. Ctrl+C to exit.")
while True:
    p.stepSimulation()
    time.sleep(1.0 / 240.0)
