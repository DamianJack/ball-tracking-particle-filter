#-------------------
# AUTHORS
# DAMIAN - 10012545
# HARSHA - 10012956
#-------------------

# n-ball trajectory simulation

import numpy as np

class Ball:
    def __init__(self, x: float, y: float, vx: float, vy: float):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
    
    def to_list(self):
        return [self.x, self.y, self.vx, self.vy]
    
    def __repr__(self):
        return f"Ball({self.x}, {self.y}, {self.vx}, {self.vy})"

class BallTrajectory:

    def __init__(self, initial_position: list, speed: float, angle_degrees: float, gravity: float = -9.81):

        # Initialize position and velocity based on input parameters
        angle      = np.deg2rad(angle_degrees)
        vx         = speed * np.cos(angle)
        vy         = speed * np.sin(angle)
        self.ball  = Ball(initial_position[0], initial_position[1], vx, vy)
        self.gravity = gravity

    def update(self, dt):
        # Update velocity and position based on gravity and time step
        self.ball.vy += self.gravity * dt
        self.ball.x  += self.ball.vx * dt
        self.ball.y  += self.ball.vy * dt

    def get_position(self):
        return [self.ball.x, self.ball.y]

    def get_velocity(self):
        return [self.ball.vx, self.ball.vy]

def simulate_ball(initial_position, speed, angle_degrees, dt=0.1) -> list:

    ball       = BallTrajectory(initial_position, speed, angle_degrees)
    t          = 0.0
    trajectory = []

    while ball.get_position()[1] >= 0:
        pos = ball.get_position().copy()
        trajectory.append((float(t), float(pos[0]), float(pos[1])))
        ball.update(dt)
        t += dt

    return trajectory

def simulate_n_balls(launches, dt=0.1) -> list:
    return [
        simulate_ball(
            launch["initial_position"],
            launch["speed"],
            launch["angle_degrees"],
            dt=dt,
        )
        for launch in launches
    ]

#======================================================================================================================================================================================

# Example usage:
if __name__ == "__main__":
    
    from plotting import BallTrajectoryPlot

    launches = [
        {"initial_position": [0, 0], "speed": 50, "angle_degrees": 45}
    ]

    trajectories = simulate_n_balls(launches)
    for i, traj in enumerate(trajectories):
        print(f"Trajectory {i+1}:")
        for pos in traj:
            print(pos)

    plotter = BallTrajectoryPlot(launches)
    plotter.visualize()

