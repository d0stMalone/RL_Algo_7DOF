import gym
import numpy as np
import pygame
import time
from gym import spaces
from envs.pybullet_self_collision_checker import SelfCollisionChecker  # Assumed to be available

# Collision and evaluation thresholds
COLLISION_THRESHOLD_LINK = 15.0
COLLISION_THRESHOLD_BASE = 20.0
COLLISION_THRESHOLD_JOINT = 10.0
EE_COLLISION_THRESHOLD = 15.0
TARGET_CRASH_THRESHOLD = 20.0
GOAL_PROXIMITY_THRESHOLD = 0.2

class SixDOFRobotEnv(gym.Env):
    def __init__(self, render_mode=False):
        super(SixDOFRobotEnv, self).__init__()

        self.num_joints = 6
        self.link_lengths = [0.5] * self.num_joints
        self.joint_limits = [(-np.pi, np.pi)] * self.num_joints

        # Curriculum parameters
        self.curriculum_scale = 0.5  # initial scale factor (range 0.1 to 1.0)
        self.last_episode_success = None

        self.max_steps = 300
        self.current_step = 0

        # Initialize state: joint angles, joint velocities, and goal position.
        self.joint_angles = np.zeros(self.num_joints)
        self.prev_joint_angles = np.zeros(self.num_joints)
        self.joint_velocities = np.zeros(self.num_joints)
        self.goal_pos = np.zeros(2)
        self.end_effector_pos = np.zeros(2)
        self.dt = 0.1  # time step for velocity estimation

        # Observation space: 6 joint angles, 6 joint velocities, and 2 goal coordinates.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.num_joints * 2 + 2,),
                                            dtype=np.float32)
        self.action_space = spaces.Box(low=-0.05, high=0.05,
                                       shape=(self.num_joints,), dtype=np.float32)

        self.render_mode = render_mode
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((600, 600))
            self.clock = pygame.time.Clock()

        self.collision_checker = SelfCollisionChecker(self.link_lengths)

    def reset(self):
        # Adjust curriculum based on previous episode performance.
        if self.last_episode_success is not None:
            if self.last_episode_success:
                self.curriculum_scale = min(self.curriculum_scale + 0.05, 1.0)
            else:
                self.curriculum_scale = max(self.curriculum_scale - 0.05, 0.1)
        self.last_episode_success = None

        self.current_step = 0

        # Initialize joint angles with small random deviations.
        self.joint_angles = np.random.uniform(low=-0.1, high=0.1, size=(self.num_joints,))
        self.prev_joint_angles = np.copy(self.joint_angles)
        self.joint_velocities = np.zeros(self.num_joints)

        # Set the goal based on the current curriculum scale.
        max_reach = 2.5 * self.curriculum_scale
        angle = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(1.0, max(1.5, max_reach))
        self.goal_pos = np.array([r * np.cos(angle), r * np.sin(angle)])

        self.update_end_effector()
        return self._get_obs()

    def step(self, action):
        self.current_step += 1
        prev_dist = np.linalg.norm(self.end_effector_pos - self.goal_pos)

        # Update joint angles and compute velocities.
        new_joint_angles = self.joint_angles + action
        new_joint_angles = np.clip(new_joint_angles, -np.pi, np.pi)
        self.joint_velocities = (new_joint_angles - self.joint_angles) / self.dt
        self.prev_joint_angles = np.copy(self.joint_angles)
        self.joint_angles = new_joint_angles

        self.update_end_effector()
        new_dist = np.linalg.norm(self.end_effector_pos - self.goal_pos)
        improvement = prev_dist - new_dist
        reward = self.compute_reward(prev_dist, new_dist, action, improvement)
        done = False

        # Check for successful goal reach (and safety).
        if new_dist < GOAL_PROXIMITY_THRESHOLD and not self.check_full_collision():
            print("[Step] ✅ Goal reached safely!")
            reward += 1000.0  # Increased bonus for goal reach
            done = True
            self.last_episode_success = True

        # Check for collision.
        elif self.check_full_collision():
            print("[Step] ❌ Collision detected!")
            reward -= 300.0  # Reduced collision penalty
            done = True
            self.last_episode_success = False
        else:
            reward += 10.0  # bonus for safe progress

        # Penalize intermediate link collisions with the target.
        origin = np.array([300.0, 300.0])
        target_screen = origin + self.goal_pos * 100
        for link_pos in self.get_link_positions()[1:-1]:
            if np.linalg.norm(link_pos - target_screen) < TARGET_CRASH_THRESHOLD:
                print("[Step] ⚠️ Intermediate link crashing into target!")
                reward -= 75.0
                done = True
                self.last_episode_success = False
                break

        if self.current_step >= self.max_steps:
            print("[Step] ⏹️ Max steps reached.")
            done = True
            self.last_episode_success = False

        reward = np.clip(reward, -300.0, 600.0)
        return self._get_obs(), reward, done, {"curriculum_scale": self.curriculum_scale}

    def compute_reward(self, prev_dist, new_dist, action, improvement):
        max_reach = np.sum(self.link_lengths)
        normalized_improvement = improvement / max_reach

        # Reward based on exponential decay of distance.
        distance_reward = np.exp(-new_dist) * 100

        # Simulate next state using forward kinematics.
        sim_joint_angles = self.joint_angles + action
        sim_joint_angles = np.clip(sim_joint_angles, -np.pi, np.pi)
        sim_positions = self.compute_forward_kinematics(sim_joint_angles)
        sim_pos = sim_positions[-1]
        sim_dir = sim_pos - self.end_effector_pos
        goal_dir = self.goal_pos - self.end_effector_pos
        sim_norm = np.linalg.norm(sim_dir) + 1e-8
        goal_norm = np.linalg.norm(goal_dir) + 1e-8
        alignment = np.dot(sim_dir, goal_dir) / (sim_norm * goal_norm)
        alignment_bonus = 20.0 * np.tanh(alignment)

        # Penalties for large actions, stagnation, and time.
        smoothness_penalty = -5.0 * np.linalg.norm(action)
        stagnation_penalty = -15.0 if improvement < 1e-4 else 0.0
        time_penalty = -1.0

        reward_total = (distance_reward +
                        normalized_improvement * 100 +
                        alignment_bonus +
                        smoothness_penalty +
                        stagnation_penalty +
                        time_penalty)
        return reward_total

    def compute_forward_kinematics(self, joint_angles):
        """Compute positions for all joints given a set of joint angles."""
        pos = np.array([0.0, 0.0])
        angle = 0.0
        positions = [pos.copy()]
        for i in range(self.num_joints):
            angle += joint_angles[i]
            dx = self.link_lengths[i] * np.cos(angle)
            dy = self.link_lengths[i] * np.sin(angle)
            pos += np.array([dx, dy])
            positions.append(pos.copy())
        return positions

    def update_end_effector(self):
        positions = self.compute_forward_kinematics(self.joint_angles)
        self.end_effector_pos = positions[-1]

    def get_link_positions(self):
        origin = np.array([300.0, 300.0])
        positions = self.compute_forward_kinematics(self.joint_angles)
        # Scale positions for rendering (using a factor of 100).
        scaled_positions = [origin + pos * 100 for pos in positions]
        return scaled_positions

    def check_full_collision(self):
        positions = self.get_link_positions()
        # Check non-adjacent link collisions.
        for i in range(len(positions) - 1):
            for j in range(i + 2, len(positions) - 1):
                if np.linalg.norm(positions[i] - positions[j]) < COLLISION_THRESHOLD_LINK:
                    print(f"[Collision] Link {i} too close to Link {j}")
                    return True
        # Check end-effector against base.
        if np.linalg.norm(positions[-1] - positions[0]) < COLLISION_THRESHOLD_BASE:
            print("[Collision] End-effector too close to base")
            return True
        # Check joints too close to base.
        for i in range(1, len(positions) - 1):
            if np.linalg.norm(positions[i] - positions[0]) < COLLISION_THRESHOLD_JOINT:
                print(f"[Collision] Joint {i} too close to base")
                return True
        # Check end-effector with intermediate links.
        for i in range(len(positions) - 2):
            if np.linalg.norm(positions[-1] - positions[i]) < EE_COLLISION_THRESHOLD:
                print(f"[Collision] End-effector too close to Link/Joint {i}")
                return True

        return self.collision_checker.check_collision(self.joint_angles)

    def render(self, mode="human"):
        if not self.render_mode:
            return

        pygame.event.pump()
        self.screen.fill((255, 255, 255))
        origin = np.array([300.0, 300.0])
        pos = origin.copy()
        link_positions = [origin.copy()]

        for i in range(self.num_joints):
            angle = np.sum(self.joint_angles[:i + 1])
            dx = self.link_lengths[i] * 100 * np.cos(angle)
            dy = self.link_lengths[i] * 100 * np.sin(angle)
            new_pos = pos + np.array([dx, dy])
            pygame.draw.line(self.screen, (0, 0, 0), pos, new_pos, 4)
            pos = new_pos
            link_positions.append(pos.copy())

        for pt in link_positions[:-1]:
            pygame.draw.circle(self.screen, (0, 0, 0), pt.astype(int), 6)
        pygame.draw.circle(self.screen, (255, 0, 0), (origin + self.goal_pos * 100).astype(int), 8)
        pygame.draw.circle(self.screen, (255, 204, 0), link_positions[-1].astype(int), 10)

        pygame.display.flip()
        self.clock.tick(30)
        time.sleep(0.01)

    def close(self):
        if self.render_mode:
            pygame.quit()

    def _get_obs(self):
        # Concatenate joint angles, joint velocities, and goal position.
        return np.concatenate([self.joint_angles, self.joint_velocities, self.goal_pos])
