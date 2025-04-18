import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time

class UR5ReachEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.num_joints = 6
        self.max_steps = 200
        self.step_count = 0

        self.robot_id = None
        self.goal_id = None
        self.goal_pos = None

        self.physics_client = p.connect(p.GUI if self.render_mode == "human" else p.DIRECT, options="--opengl2")
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_joints + 3,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)

        p.loadURDF("plane.urdf")

        # UR5 model
        self.robot_id = p.loadURDF("ur5/ur5.urdf", useFixedBase=True)

        for i in range(self.num_joints):
            p.resetJointState(self.robot_id, i, targetValue=np.random.uniform(-0.1, 0.1))

        self.goal_pos = np.array([
            np.random.uniform(0.4, 0.7),
            np.random.uniform(-0.2, 0.2),
            np.random.uniform(0.3, 0.6)
        ])

        visual_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02]*3, rgbaColor=[1, 0, 0, 1])
        collision_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02]*3)
        self.goal_id = p.createMultiBody(0, collision_shape_id, visual_shape_id, self.goal_pos)

        p.resetDebugVisualizerCamera(1.4, 111, -26, [0.5, 0, 0.4])
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        scaled_action = action * (np.pi / 2)

        for i in range(self.num_joints):
            p.setJointMotorControl2(
                self.robot_id, i, p.POSITION_CONTROL, targetPosition=scaled_action[i], force=500
            )

        p.stepSimulation()
        if self.render_mode == "human":
            time.sleep(1.0 / 60.0)

        ee_state = p.getLinkState(self.robot_id, 7, computeLinkVelocity=1)  # UR5 EE = link 7
        ee_pos = np.array(ee_state[0])
        ee_vel = np.linalg.norm(ee_state[6])
        distance = np.linalg.norm(ee_pos - self.goal_pos)

        p.addUserDebugLine(self.goal_pos, ee_pos, [0, 1, 0], 2, lifeTime=0.1)

        reward = -distance**2 + 5.0 * np.exp(-20 * distance) - 0.1 * ee_vel - 0.01 * self.step_count
        if distance < 0.02:
            reward += 3.0

        terminated = distance < 0.02
        truncated = self.step_count >= self.max_steps
        info = {"distance": distance}
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        joint_states = [p.getJointState(self.robot_id, i)[0] for i in range(self.num_joints)]
        ee_pos = np.array(p.getLinkState(self.robot_id, 7)[0])
        return np.concatenate([joint_states, ee_pos], dtype=np.float32)

    def render(self): pass

    def close(self):
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None
