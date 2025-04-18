import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import os

class KukaReachEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, use_dijkstra=False):
        super().__init__()
        self.render_mode = render_mode
        self.use_dijkstra = use_dijkstra
        self.max_steps = 200
        self.step_count = 0

        self.num_joints = 7
        self.joint_indices = list(range(self.num_joints))
        self.ee_link_index = 6

        self.robot_id = None
        self.goal_id = None
        self.ee_marker_id = None
        self.goal_pos = None

        self.physics_client = p.connect(p.GUI if self.render_mode == "human" else p.DIRECT, options="--opengl2")
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_joints + 3,), dtype=np.float32)

    def get_ee_position(self):
        ee_state = p.getLinkState(self.robot_id, self.ee_link_index)
        return np.array(ee_state[0])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

        urdf_path = os.path.join(pybullet_data.getDataPath(), "kuka_iiwa/model.urdf")
        self.robot_id = p.loadURDF(urdf_path, useFixedBase=True)

        for i in self.joint_indices:
            p.resetJointState(self.robot_id, i, np.random.uniform(-0.1, 0.1))

        # Goal range slightly limited (easy → hard)
        self.goal_pos = np.array([
            np.random.uniform(0.5, 0.6),
            np.random.uniform(-0.1, 0.1),
            np.random.uniform(0.3, 0.5)
        ])

        visual_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02]*3, rgbaColor=[1, 0, 0, 1])
        collision_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02]*3)
        self.goal_id = p.createMultiBody(0, collision_id, visual_id, self.goal_pos)

        # Green sphere marker on EE
        ee_pos = self.get_ee_position()
        ee_marker_offset = [ee_pos[0], ee_pos[1], ee_pos[2] + 0.02]
        ee_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[0, 1, 0, 1])
        self.ee_marker_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=ee_shape, basePosition=ee_marker_offset)

        # Camera setup
        p.resetDebugVisualizerCamera(1.2, 135, -30, [0.5, 0, 0.4])

        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1

        # Action smoothing
        max_delta = 0.05
        delta_angles = max_delta * np.tanh(np.clip(action, -1, 1))

        current_angles = [p.getJointState(self.robot_id, i)[0] for i in self.joint_indices]
        new_angles = np.array(current_angles) + delta_angles

        for i in self.joint_indices:
            joint_info = p.getJointInfo(self.robot_id, i)
            low, high = joint_info[8], joint_info[9]
            new_angles[i] = np.clip(new_angles[i], low, high)

            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=new_angles[i],
                force=500
            )

        for _ in range(5):
            p.stepSimulation()

        obs = self._get_obs()
        ee_pos = self.get_ee_position()
        distance = np.linalg.norm(ee_pos - self.goal_pos)

        # Reward shaping
        reward = -distance
        if distance < 0.03:
            reward += 10
        elif distance < 0.05:
            reward += 3
        elif distance < 0.08:
            reward += 1

        done = distance < 0.03 or self.step_count >= self.max_steps

        # Update EE marker
        p.resetBasePositionAndOrientation(self.ee_marker_id, ee_pos, [0, 0, 0, 1])

        return obs, reward, done, False, {"distance": distance}

    def _get_obs(self):
        joint_states = [p.getJointState(self.robot_id, i)[0] for i in self.joint_indices]
        ee_pos = self.get_ee_position()
        return np.concatenate([joint_states, ee_pos], dtype=np.float32)

    def render(self): pass

    def close(self):
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None
