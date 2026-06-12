#-------------------
# AUTHORS
# DAMIAN - 10012545
# HARSHA - 10012956
#-------------------

import numpy as np
from ball_trajectory import BallTrajectory, simulate_ball, simulate_n_balls

class BallObservation:

    def __init__(self, true_position: list, noise_stddev: float, dropout_prob: float = 0.1):
        self.true_position = np.array(true_position, dtype=float)
        self.noise_stddev  = noise_stddev
        self.dropout_prob  = dropout_prob

    def observe(self):
        if np.random.rand() < self.dropout_prob:
            return []
        noise = np.random.normal(0, self.noise_stddev, size=2)
        print(noise)
        return self.true_position + noise

def simulate_observations(launches, noise_stddev, dropout_prob: float = 0.1, dt=0.1):
    trajectories = simulate_n_balls(launches, dt=dt)
    observations = []

    for trajectory in trajectories:
        obs = []
        for t, x, y in trajectory:
            ball_obs = BallObservation([x, y], noise_stddev, dropout_prob=dropout_prob)
            obs.append((t, *ball_obs.observe()))
        observations.append(obs)

    return observations

# Example usage:
if __name__ == "__main__":
    launches = [
        {"initial_position": [0, 0], "speed": 50, "angle_degrees": 45},
        {"initial_position": [0, 0], "speed": 60, "angle_degrees": 30},
    ]
    noise_stddev = 5.0
    dropout_prob = 0.2
    observations = simulate_observations(launches, noise_stddev, dropout_prob=dropout_prob)
    for obs in observations:
        for item in obs:
            if len(item) == 3:
                t, x_obs, y_obs = item
                print(f"Time: {t:.2f} s, Observed Position: ({x_obs:.2f}, {y_obs:.2f})")
            else:
                t = item[0]                
                print(f"Time: {t:.2f} s, Observation dropped")
    