import numpy as np
import pybullet as p
import time
from collections import deque

# Coarse discretization for speed
JOINT_RESOLUTION = 0.3
GOAL_THRESHOLD = 0.05  # meters

def get_neighbors(config):
    """Generate neighboring joint configurations"""
    neighbors = []
    for i in range(len(config)):
        for delta in [-JOINT_RESOLUTION, JOINT_RESOLUTION]:
            new_config = list(config)
            new_config[i] += delta
            neighbors.append(new_config)
    return neighbors

def fk(joint_angles, robot_id=0, ee_link=6):
    """Compute end-effector position using FK"""
    for i, angle in enumerate(joint_angles):
        p.resetJointState(robot_id, i, angle)
    ee_state = p.getLinkState(robot_id, ee_link)
    return np.array(ee_state[0])

def dijkstra_plan(robot_id, start_config, goal_pos):
    start_time = time.time()
    node_count = 0

    open_list = deque()
    open_list.append((start_config, 0))
    visited = set()
    came_from = {}

    rounded_start = tuple(np.round(start_config, 2))
    visited.add(rounded_start)

    while open_list:
        current_config, cost = open_list.popleft()
        node_count += 1

        if node_count % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"🔄 Explored {node_count:,} configs | Elapsed: {elapsed:.2f}s")

        ee_pos = fk(current_config, robot_id)

        if np.linalg.norm(ee_pos - goal_pos) < GOAL_THRESHOLD:
            print("🎯 Goal reached!")
            path = [current_config]
            while tuple(np.round(current_config, 2)) in came_from:
                current_config = came_from[tuple(np.round(current_config, 2))]
                path.append(current_config)
            path.reverse()
            total_time = time.time() - start_time
            print(f"\n✅ Dijkstra completed.")
            print(f"🧠 Nodes expanded: {node_count:,}")
            print(f"⏱️ Total time: {total_time:.2f} seconds\n")
            return path

        for neighbor in get_neighbors(current_config):
            rounded = tuple(np.round(neighbor, 2))
            if rounded in visited:
                continue
            visited.add(rounded)
            open_list.append((neighbor, cost + 1))
            came_from[rounded] = current_config

    print("❌ No path found.")
    print(f"🧠 Nodes explored: {node_count:,}")
    print(f"⏱️ Time: {time.time() - start_time:.2f}s")
    return None
