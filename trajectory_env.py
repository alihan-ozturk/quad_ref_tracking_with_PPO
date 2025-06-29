import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.spatial.transform import Rotation

from drone_physics import DronePhysics

class ReferenceTrajectory:
    def __init__(self, dt=0.02):
        self.dt = dt
        self.amplitude = np.array([5, 5, 2.5])
        self.frequency = np.array([0.1, 0.15, 0.2])
        self.phase = np.array([0, np.pi/2, 0])
        self.center = np.array([0, 0, 5])

    def get_state(self, t):
        angle = 2 * np.pi * self.frequency * t + self.phase
        pos = self.center + self.amplitude * np.sin(angle)
        
        vel = 2 * np.pi * self.frequency * self.amplitude * np.cos(angle)
        
        return pos, vel

class DroneTrajectoryEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 50}
    
    def __init__(self, max_steps=1000):
        super().__init__()
        self.drone = DronePhysics()
        self.trajectory = ReferenceTrajectory(dt=self.drone.DT)
        self.max_steps = max_steps
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.time = 0.0
        
        start_pos, _ = self.trajectory.get_state(0)
        self.drone.reset(pos=start_pos)
        
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        self.time += self.drone.DT
        
        crashed = self.drone.step(action)
        
        obs = self._get_obs()
        reward, pos_err_val = self._compute_reward()
        
        # Sonlanma koşulları
        terminated = crashed or (self.current_step >= self.max_steps)
        if pos_err_val > 8.0:
             terminated = True
             reward -= 2.0

        truncated = False
        
        info = {'pos_error': pos_err_val}
        
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        ref_pos, ref_vel = self.trajectory.get_state(self.time)
        
        pos_error = ref_pos - self.drone.pos
        vel_error = ref_vel - self.drone.vel
        
        obs = np.concatenate([
            pos_error,
            vel_error,
            self.drone.quat.as_quat(),
            self.drone.omega
        ]).astype(np.float32)
        return obs

    def _compute_reward(self):
        ref_pos, _ = self.trajectory.get_state(self.time)
        
        pos_error_norm = np.linalg.norm(ref_pos - self.drone.pos)
        pos_reward = np.exp(-1.5 * pos_error_norm**2)
        
        up_vec = self.drone.quat.apply([0, 0, 1])
        orientation_reward = np.clip(up_vec[2], 0, 1)**2
        

        control_penalty = 0.01 * np.mean(np.square(self.drone.omega))
        
        reward = 1.5 * pos_reward + 0.5 * orientation_reward - control_penalty
        
        return reward, pos_error_norm