import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

class DroneAnimation:
    def __init__(self, drone_history, ref_history, trail_length=150):
        self.drone_history = drone_history
        self.ref_history = ref_history
        self.trail_length = trail_length
        self.num_frames = len(drone_history['pos'])

        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.rotor_colors = ['red', 'blue', 'green', 'yellow']

    def _animate(self, i):
        start_index = max(0, i - self.trail_length)
        current_trail = slice(start_index, i + 1)
        
        pos = self.drone_history['pos'][i]
        rotor_pos = self.drone_history['rotors'][i]
        ref_pos = self.ref_history['pos'][i]

        self.ax.clear()
        
        self.ax.set_xlim(-8, 8); self.ax.set_ylim(-8, 8); self.ax.set_zlim(0, 10)
        self.ax.set_xlabel('X'); self.ax.set_ylabel('Y'); self.ax.set_zlabel('Z')
        self.ax.set_title(f"Takip Anismasyonu - Adım: {i}")

        ref_path = np.array(self.ref_history['pos'])
        self.ax.plot(ref_path[:, 0], ref_path[:, 1], ref_path[:, 2], 'r--', label='Referans Yörünge')
        
        drone_trail = np.array(self.drone_history['pos'][current_trail])
        self.ax.plot(drone_trail[:, 0], drone_trail[:, 1], drone_trail[:, 2], 'b-', alpha=0.7, label='Drone Yörüngesi')
        
        self.ax.scatter(ref_pos[0], ref_pos[1], ref_pos[2], c='red', marker='x', s=100)
        
        for r_pos in rotor_pos:
            self.ax.plot([pos[0], r_pos[0]], [pos[1], r_pos[1]], [pos[2], r_pos[2]], 'k-', lw=1.5)
        self.ax.scatter(rotor_pos[:, 0], rotor_pos[:, 1], rotor_pos[:, 2], c=self.rotor_colors, s=30)
        
        if i == 0: self.ax.legend()

        if i == self.num_frames - 1:
            self.fig.canvas.start_event_loop(2)
            plt.close(self.fig)

    def run(self):
        ani = FuncAnimation(self.fig, self._animate, frames=self.num_frames, interval=20, repeat=False)
        plt.show()