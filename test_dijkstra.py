# test_dijkstra_only.py
import pybullet as p
import gymnasium as gym
from envs.dijkstra_planner import dijkstra_plan
from envs.kuka_reach_env import KukaReachEnv
from gymnasium.envs.registration import register

register(
    id="KukaReach6DOF-v0",
    entry_point="envs.kuka_reach_env:KukaReachEnv"
)

env = gym.make("KukaReach6DOF-v0", render_mode="human", use_dijkstra=True)
obs, _ = env.reset()

robot_id = env.robot_id
goal = env.goal_pos
start_angles = [p.getJointState(robot_id, i)[0] for i in range(6)]

path = dijkstra_plan(robot_id, start_angles, goal)

if path:
    print(f"✅ Dijkstra path found with {len(path)} waypoints")
    for i, angles in enumerate(path):
        for j in range(6):
            p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL, angles[j])
        for _ in range(10):
            p.stepSimulation()
else:
    print("❌ Dijkstra failed to find path")

while True:
    p.stepSimulation()
