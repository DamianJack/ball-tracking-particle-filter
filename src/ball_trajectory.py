#-------------------
# AUTHORS
# DAMIAN - 10012545
# HARSHA - 10012956
#-------------------

# n-ball trajectory simulation

import numpy as np

class BallTrajectory:

    def __init__(self, initial_position: list, speed: float, angle_degrees: float, gravity: float = -9.81):

        # Initialize position and velocity based on input parameters
        self.pos     = np.array(initial_position, dtype=float)
        angle        = np.deg2rad(angle_degrees)
        self.vel     = np.array([speed * np.cos(angle), speed * np.sin(angle)], dtype=float)
        self.gravity = gravity

    def update(self, dt):
        # Update velocity and position based on gravity and time step
        self.vel[1] += self.gravity * dt
        self.pos    += self.vel * dt

    def get_position(self):
        return self.pos

    def get_velocity(self):
        return self.vel

def simulate_ball(initial_position, speed, angle_degrees, dt=0.1):

    ball       = BallTrajectory(initial_position, speed, angle_degrees)
    t          = 0.0
    trajectory = []

    while ball.get_position()[1] >= 0:
        pos = ball.get_position().copy()
        trajectory.append((float(t), float(pos[0]), float(pos[1])))
        ball.update(dt)
        t += dt

    return trajectory

def simulate_n_balls(launches, dt=0.1):
    return [
        simulate_ball(
            launch["initial_position"],
            launch["speed"],
            launch["angle_degrees"],
            dt=dt,
        )
        for launch in launches
    ]
    
# Example usage:
if __name__ == "__main__":

    launches = [
        {"initial_position": [0, 0], "speed": 50, "angle_degrees": 45},
        {"initial_position": [0, 0], "speed": 60, "angle_degrees": 30},
    ]

    trajectories = simulate_n_balls(launches)
    for i, traj in enumerate(trajectories):
        print(f"Trajectory {i+1}:")
        for pos in traj:
            print(pos)