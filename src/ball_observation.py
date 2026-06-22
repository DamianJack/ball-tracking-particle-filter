#-------------------
# AUTHORS
# DAMIAN - 10012545
# HARSHA - 10012956
#-------------------

import numpy as np
from ball_trajectory import simulate_n_balls

class BallObservation:
    def __init__(self, trajectory, noise_stddev: float, dropout_prob: float, rng=None):

        self.trajectory     = trajectory
        self.noise_stddev   = noise_stddev
        self.dropout_prob   = dropout_prob
        self.rng            = rng if rng is not None else np.random.default_rng()

    def is_dropped(self, t):
        return self.rng.random() < self.dropout_prob

    def observe_position(self, true_position):
        noise = self.rng.normal(0.0, self.noise_stddev, size=2)
        return np.array(true_position, dtype=float) + noise

    def simulate_observations(self):
        observations = []

        for trajectory_point in self.trajectory:
            t = trajectory_point[0]
            if self.is_dropped(t):
                observations.append((t, None))
            else:
                true_position = trajectory_point[1:]
                observed = self.observe_position(true_position)
                if observed[1] < 0:
                    observations.append((t, None))  # below-ground reading treated as dropout
                else:
                    observations.append((t, observed))

        return observations

#======================================================================================================================================================================================

# Example usage:
if __name__ == "__main__":

    launches = [
        {"initial_position": [0, 0], "speed": 50, "angle_degrees": 45},
        {"initial_position": [0, 0], "speed": 60, "angle_degrees": 30},
    ]

    trajectories      = simulate_n_balls(launches, dt=0.1)

    observers = [
                BallObservation(traj, noise_stddev=3.0, dropout_prob=0.2)
                for traj in trajectories
                ]

    for i, obs in enumerate(observers):
        print(f"Ball {i + 1}:")
        observations = obs.simulate_observations()
        for obs in observations:
            if obs[1] is not None:
                print(f"Time: {obs[0]:.2f} s, Observed Position: {obs[1][0]:.2f}, {obs[1][1]:.2f}")
            else:
                print(f"Time: {obs[0]:.2f} s, Observation Dropped")