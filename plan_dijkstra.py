import gymnasium as gym
import pybullet as p
import numpy as np
from gymnasium.envs.registration import register
from envs.kuka_reach_env import KukaReachEnv
from envs.dijkstra_planner import dijkstra_plan
from stable_baselines3.common.vec_env import VecNormalize

# Register the env
register(
    id="KukaReach6DOF-v0",
    entry_point="envs.kuka_reach_env:KukaReachEnv"
)

# Create env
env = gym.make("KukaReach6DOF-v0", render_mode="human")
obs, _ = env.reset()

# Get robot ID and goal
robot_id = env.robot_id
goal_pos = env.goal_pos

# Get current joint state
start_angles = [p.getJointState(robot_id, i)[0] for i in range(6)]

# Run Dijkstra planner
print("🧭 Running Dijkstra planner...")
path = dijkstra_plan(robot_id, start_angles, goal_pos)

# Execute path
if path:
    print(f"✅ Planned path with {len(path)} waypoints.")
    for idx, angles in enumerate(path):
        for i, angle in enumerate(angles):
            p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, targetPosition=angle)
        for _ in range(10):  # Let joints settle
            p.stepSimulation()
        print(f"  ↪ Executed waypoint {idx+1}/{len(path)}")
else:
    print("❌ Dijkstra failed to find a path.")

# Hold GUI open
while True:
    p.stepSimulation()
