#-------------------
# AUTHORS
# DAMIAN - 10012545
# HARSHA - 10012956
#-------------------

import numpy as np

from ball_observation import BallObservation
from ball_trajectory import Ball, simulate_n_balls

class Particle(Ball):
    def __init__(self, x: float, y: float, vx: float, vy: float):
        super().__init__(x, y, vx, vy)

    def to_list(self):
        return [self.x, self.y, self.vx, self.vy]


class ParticleFilter:
    def __init__(self, num_particles: int, ball_observation: BallObservation,
                 area: float = 100.0, noise: float = 0.2, gravity: float = -9.81):

        self.num_particles    = num_particles
        self.ball_observation = ball_observation
        self.area             = area
        self.noise            = noise
        self.gravity          = gravity
        self.rng              = np.random.default_rng()
        self.initialize_particles()

    # ------------------------------------------------------------------
    # INITIALISE
    # ------------------------------------------------------------------
    def initialize_particles(self):
        self.x  = self.rng.uniform(-self.area/2, self.area/2, self.num_particles)
        self.y  = self.rng.uniform(0, self.area/2, self.num_particles)

        speed   = self.rng.uniform(0.0, 60.0, self.num_particles)
        angle   = self.rng.uniform(0.0, np.pi, self.num_particles)  # upward launches only

        self.vx = speed * np.cos(angle)
        self.vy = speed * np.sin(angle)

        self.weights = np.ones(self.num_particles) / self.num_particles
        self._sync_particles()

    def _sync_particles(self):
        """Keep self.particles (N×4 array) in sync with the four component arrays."""
        self.particles = np.column_stack((self.x, self.y, self.vx, self.vy))

    # ------------------------------------------------------------------
    # PREDICT  –  projectile motion + process noise
    # ------------------------------------------------------------------
    def predict(self, dt: float):
        # 1. Physics step (deterministic)
        self.x  += self.vx * dt
        self.y  += self.vy * dt
        self.vy += self.gravity * dt

        # 2. Process noise (consistent RNG)
        self.x  += self.rng.normal(0, self.noise,       self.num_particles)
        self.y  += self.rng.normal(0, self.noise,       self.num_particles)
        self.vx += self.rng.normal(0, self.noise * 0.5, self.num_particles)
        self.vy += self.rng.normal(0, self.noise * 0.5, self.num_particles)

        # 3. Ground clamp: particles cannot exist below y=0
        grounded          = self.y < 0
        self.y[grounded]  = 0.0
        self.vy[grounded] = 0.0

        self._sync_particles()
        return self.particles

    # ------------------------------------------------------------------
    # LIKELIHOOD  –  how well does each particle match the observation?
    # ------------------------------------------------------------------
    def likelihood(self, observation):
        obs_x, obs_y = observation
        distances    = np.sqrt((self.x - obs_x)**2 + (self.y - obs_y)**2)

        sigma = self.ball_observation.noise_stddev

        likelihoods  = np.exp(-0.5 * (distances / sigma)**2)
        likelihoods += 1e-300          # prevent all-zero weights

        return likelihoods / likelihoods.sum()

    # ------------------------------------------------------------------
    # RESAMPLE  –  multinomial resampling (only when weight diversity is low)
    # ------------------------------------------------------------------
    def resample(self):
        neff = 1.0 / np.sum(np.square(self.weights))
        if neff >= self.num_particles / 2:
            return  # particles are well-spread; skip resampling
        indices      = self.rng.choice(self.num_particles, size=self.num_particles,
                                      replace=True, p=self.weights)
        self.x       = self.x[indices]
        self.y       = self.y[indices]
        self.vx      = self.vx[indices]
        self.vy      = self.vy[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles
        self._sync_particles()

    # ------------------------------------------------------------------
    # UPDATE  –  weight and resample in one call
    # ------------------------------------------------------------------
    def update(self, observation):
        if observation is not None:
            self.weights = self.likelihood(observation)
        self.resample()

    # ------------------------------------------------------------------
    # ESTIMATE  –  return the weighted mean state
    # ------------------------------------------------------------------
    def estimate(self):
        w = self.weights
        return np.array([
            np.average(self.x,  weights=w),
            np.average(self.y,  weights=w),
            np.average(self.vx, weights=w),
            np.average(self.vy, weights=w),
        ])

    # ------------------------------------------------------------------
    # STEP  –  one full predict→update→estimate cycle
    # ------------------------------------------------------------------
    def step(self, dt: float, observation):
        self.predict(dt)
        self.update(observation)
        return self.estimate()


# ======================================================================
# Example usage
# ======================================================================
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

    # One independent ParticleFilter per ball
    filters = [
        ParticleFilter(num_particles=1000, ball_observation=obs, area=100.0, noise=1.0)
        for obs in observers
    ]

    # Run all filters for one ball's worth of observations as a smoke-test
    observations_ball0 = observers[0].simulate_observations()
    observations_ball1 = observers[1].simulate_observations()

    for t, obs in [observations_ball0, observations_ball1][0]:  # Just test the first ball for now
        print(f"Time: {t:.2f} s, Observation: {obs}")
        estimate = filters[0].step(dt=0.1, observation=obs)
        print(f"Estimated State: x={estimate[0]:.2f}, y={estimate[1]:.2f}, "
              f"vx={estimate[2]:.2f}, vy={estimate[3]:.2f}\n")
    

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    # Plot true trajectory
    traj = trajectories[0]
    traj_x = [pos[1] for pos in traj]
    traj_y = [pos[2] for pos in traj]
    ax.plot(traj_x, traj_y, label="True Trajectory", color="blue")
    # Plot observations
    obs_x = [obs[1][0] for obs in observations_ball0 if obs[1] is not None]
    obs_y = [obs[1][1] for obs in observations_ball0 if obs[1] is not None]
    ax.scatter(obs_x, obs_y, label="Observations", color="red", s=20)
    # Plot particle filter estimates
    filters[0].initialize_particles()
    estimates_x = []
    estimates_y = []
    for t, obs in observations_ball0:
        estimate = filters[0].step(dt=0.1, observation=obs)
        estimates_x.append(estimate[0])
        estimates_y.append(estimate[1])
    ax.plot(estimates_x, estimates_y, label="Particle Filter Estimate", color="green")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_title("Particle Filter Tracking of Ball Trajectory")
    ax.legend()
    plt.show()

    

