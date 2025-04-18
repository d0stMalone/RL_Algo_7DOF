import os
import numpy as np
import pandas as pd
import pybullet as p
import pybullet_data
from envs.kuka_reach_env import KukaReachEnv
from envs.dijkstra_planner import dijkstra_plan

# Create output directory
log_dir = "dijkstra_logs"
os.makedirs(log_dir, exist_ok=True)

# Initialize environment
env = KukaReachEnv(render_mode=None)
obs, _ = env.reset()

# Get joint state and goal
start_angles = [p.getJointState(env.robot_id, i)[0] for i in env.joint_indices]
goal_pos = env.goal_pos

# Run Dijkstra planner
print("⚙️ Running Dijkstra planner...")
path = dijkstra_plan(env.robot_id, start_angles, goal_pos)

if path is None:
    print("❌ No path found.")
    env.close()
    exit()

print(f"✅ Path found with {len(path)} steps. Executing...")

# Logging
log_data = []

for step_idx, joint_config in enumerate(path):
    for i, angle in enumerate(joint_config):
        p.resetJointState(env.robot_id, i, angle)
    p.stepSimulation()

    ee_pos = env.get_ee_position()
    dist = np.linalg.norm(ee_pos - goal_pos)
    reward = -dist + (10 if dist < 0.03 else 0)

    log_data.append({
        "Step": step_idx + 1,
        "Distance": dist,
        "Reward": reward
    })

# Save trajectory CSV
trajectory_file = os.path.join(log_dir, "dijkstra_trajectory.csv")
pd.DataFrame(log_data).to_csv(trajectory_file, index=False)
print(f"📄 Dijkstra trajectory saved: {trajectory_file}")

# Save summary CSV (1-row like PPO/DDPG)
final_dist = log_data[-1]["Distance"]
total_reward = sum([entry["Reward"] for entry in log_data])
success = final_dist < 0.03

summary_df = pd.DataFrame([{
    "Episode": 1,
    "Reward": total_reward,
    "Distance": final_dist,
    "Success": success
}])
summary_file = os.path.join(log_dir, "eval_summary.csv")
summary_df.to_csv(summary_file, index=False)
print(f"📄 Summary saved: {summary_file}")

env.close()
