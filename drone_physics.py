# drone_physics.py

import numpy as np
from scipy.spatial.transform import Rotation

class DronePhysics:
    def __init__(self):
        self.DT = 0.02
        self.GRID_SIZE = 20.0
        self.MASS = 1.0
        self.I = np.array([0.01, 0.01, 0.02])
        self.ARM_LEN = 0.25
        self.K_THRUST = 3e-5
        self.K_DRAG = 1e-6
        self.B_DRAG = 0.1
        self.GRAVITY = 9.81
        self.MAX_RPM = 750.0
        self.MAX_VEL = 50.0
        self.MAX_OMEGA = 50.0
        self.reset()

    def reset(self, pos=np.zeros(3), quat=Rotation.identity()):
        self.pos = pos
        self.vel = np.zeros(3)
        self.quat = quat
        self.omega = np.zeros(3)

    def step(self, actions):
        actions = np.clip(actions, -1.0, 1.0)
        rpm = (actions + 1.0) * 0.5 * self.MAX_RPM
        thrusts = self.K_THRUST * rpm**2
        T = thrusts
        
        f_body = np.array([0, 0, np.sum(T)])
        
        tau_body = np.array([
             self.ARM_LEN * (T[1] - T[3]),
             self.ARM_LEN * (T[2] - T[0]),
             self.K_DRAG * (T[0] - T[1] + T[2] - T[3])
        ])
        
        tau_body -= 0.2 * self.omega

        f_world = self.quat.apply(f_body)
        f_world -= self.B_DRAG * self.vel
        f_world[2] -= self.MASS * self.GRAVITY
        
        accel = f_world / self.MASS
        omega_dot = (tau_body - np.cross(self.omega, self.I * self.omega)) / self.I

        self.vel += accel * self.DT
        self.pos += self.vel * self.DT
        self.omega += omega_dot * self.DT

        self.vel = np.clip(self.vel, -self.MAX_VEL, self.MAX_VEL)
        self.omega = np.clip(self.omega, -self.MAX_OMEGA, self.MAX_OMEGA)

        rotation_delta = Rotation.from_rotvec(self.omega * self.DT)
        self.quat = rotation_delta * self.quat
        
        terminated = np.any(np.abs(self.pos) > self.GRID_SIZE)
        return terminated