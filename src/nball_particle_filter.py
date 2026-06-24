#-------------------
# AUTHORS
# DAMIAN - 10012545
# HARSHA - 10012956
#-------------------

import numpy as np

from plotting import ParticleFilterTrajectoryPlot
from ball_observation import BallObservation
from ball_trajectory import simulate_n_balls
from particle_filter import ParticleFilter
from itertools import permutations

class MultiBallParticleFilter(ParticleFilter):
    def __init__(self, num_particles, n_balls, noise_stddev,
                 area=100.0, noise=0.2, gravity=-9.81):

        self.n_balls = n_balls
        self._sigma = noise_stddev
        # Pass ball_observation=None: base __init__ only stores it, never calls it,
        # and our overridden likelihood() uses self._sigma directly.
        super().__init__(num_particles, noise_stddev=noise_stddev, area=area, noise=noise, gravity=gravity)

    # --------------------------------------------------------------
    # OVERRIDE: initialize n-ball state
    # --------------------------------------------------------------
    def initialize_particles(self):
        # state: (N, n_balls, 4) — columns are [x, y, vx, vy]
        self.state = np.zeros((self.num_particles, self.n_balls, 4))

        # x, y
        self.state[:, :, 0] = self.rng.uniform(-self.area/2, self.area/2,
                                               (self.num_particles, self.n_balls))
        self.state[:, :, 1] = self.rng.uniform(0, self.area/2,
                                               (self.num_particles, self.n_balls))

        # vx, vy
        speed = self.rng.uniform(0, 60, (self.num_particles, self.n_balls))
        angle = self.rng.uniform(0, np.pi, (self.num_particles, self.n_balls))

        self.state[:, :, 2] = speed * np.cos(angle)
        self.state[:, :, 3] = speed * np.sin(angle)

        self.weights = np.ones(self.num_particles) / self.num_particles

    # --------------------------------------------------------------
    # OVERRIDE: vectorized prediction for n balls
    # --------------------------------------------------------------
    def predict(self, dt):
        self.state[:, :, 0] += self.state[:, :, 2] * dt
        self.state[:, :, 1] += self.state[:, :, 3] * dt
        self.state[:, :, 3] += self.gravity * dt

        # Match base class: smaller noise on velocity than on position
        pos_noise = self.rng.normal(0, self.noise,       (self.num_particles, self.n_balls, 2))
        vel_noise = self.rng.normal(0, self.noise * 0.5, (self.num_particles, self.n_balls, 2))
        self.state[:, :, :2] += pos_noise
        self.state[:, :, 2:] += vel_noise

        # Ground clamp
        mask = self.state[:, :, 1] < 0
        self.state[:, :, 1][mask] = 0
        self.state[:, :, 3][mask] = 0

    # --------------------------------------------------------------
    # OVERRIDE: likelihood using best-assignment over all permutations
    # observations: list of length n_balls, each entry is (x, y) or None
    # --------------------------------------------------------------
    def likelihood(self, observations):
        # Keep only real (non-dropout) observations
        obs = [o for o in observations if o is not None]
        m = len(obs)

        if m == 0:
            return np.ones(self.num_particles) / self.num_particles

        sigma = self._sigma

        # All ways to assign m observations to n_balls (order matters)
        perms = list(permutations(range(self.n_balls), m))

        logL = np.zeros(self.num_particles)

        for p in range(self.num_particles):
            balls = self.state[p, :, :2]  # (n_balls, 2)
            best = -np.inf

            for perm in perms:
                d2 = 0.0
                for j, ball_idx in enumerate(perm):
                    ox, oy = obs[j]
                    bx, by = balls[ball_idx]
                    d2 += (bx - ox)**2 + (by - oy)**2

                log_like = -0.5 * d2 / (sigma * sigma)
                if log_like > best:
                    best = log_like

            logL[p] = best

        # Numerically stable normalization
        logL -= np.max(logL)
        L = np.exp(logL)
        L += 1e-300
        return L / L.sum()

    # --------------------------------------------------------------
    # OVERRIDE: resample using self.state instead of component arrays
    # Bug fix: base class resample() uses self.x/y/vx/vy which don't exist here
    # --------------------------------------------------------------
    def resample(self):
        neff = 1.0 / np.sum(np.square(self.weights))
        if neff >= self.num_particles / 2:
            return
        indices = self.rng.choice(self.num_particles, size=self.num_particles,
                                  replace=True, p=self.weights)
        self.state = self.state[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    # --------------------------------------------------------------
    # OVERRIDE: update accepts a list of observations (one per ball)
    # Bug fix: base class update() takes a single observation, not a list
    # --------------------------------------------------------------
    def update(self, observations):
        self.weights = self.likelihood(observations)
        self.resample()

    # --------------------------------------------------------------
    # OVERRIDE: step accepts a list of observations
    # Bug fix: base class step() signature only accepts one observation
    # --------------------------------------------------------------
    def step(self, dt, observations):
        self.predict(dt)
        self.update(observations)
        return self.estimate()

    # --------------------------------------------------------------
    # OVERRIDE: estimate n ball states
    # --------------------------------------------------------------
    def estimate(self):
        w = self.weights[:, None, None]
        return np.sum(self.state * w, axis=0)   # shape (n_balls, 4)


if __name__ == "__main__":
    launches = [
        {"initial_position": [0, 0], "speed": 50, "angle_degrees": 45},
        {"initial_position": [0, 0], "speed": 60, "angle_degrees": 30},
    ]

    trajectories = simulate_n_balls(launches, dt=0.1)

    observers = [
        BallObservation(traj, noise_stddev=8.0, dropout_prob=0.2)
        for traj in trajectories
    ]

    observations_ball0 = observers[0].simulate_observations()
    observations_ball1 = observers[1].simulate_observations()

    # Multi-ball filter: tracks both balls simultaneously.
    # noise_stddev is the observation noise used in the likelihood — it must match
    # the noise used when generating observations (BallObservation.noise_stddev).
    pf = MultiBallParticleFilter(
        num_particles=1000,
        n_balls=2,
        noise_stddev=8.0,
        area=100.0,
        noise=1.0
    )

    estimates = []  # list of (n_balls, 4) arrays
    for (t0, obs0), (t1, obs1) in zip(observations_ball0, observations_ball1):
        obs_list = [obs0, obs1]
        estimate = pf.step(dt=0.1, observations=obs_list)
        estimates.append(estimate)
        print(f"Time: {t0:.2f} s")
        print(f"  Ball 0 — x={estimate[0,0]:.2f}, y={estimate[0,1]:.2f}, "
              f"vx={estimate[0,2]:.2f}, vy={estimate[0,3]:.2f}")
        print(f"  Ball 1 — x={estimate[1,0]:.2f}, y={estimate[1,1]:.2f}, "
              f"vx={estimate[1,2]:.2f}, vy={estimate[1,3]:.2f}\n")

    # Plot each ball separately
    for i, (traj, obs_seq) in enumerate(zip(trajectories,
                                            [observations_ball0, observations_ball1])):
        ball_estimates = [e[i] for e in estimates]
        plotter = ParticleFilterTrajectoryPlot(
            trajectory=traj,
            observations=obs_seq,
            estimates=ball_estimates
        )
        plotter.visualize()
