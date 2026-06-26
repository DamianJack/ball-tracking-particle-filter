#-------------------
# AUTHORS
# DAMIAN - 10012545
# HARSHA - 10012956
#-------------------

import numpy as np

from plotting import ParticleFilterTrajectoryPlot
from ball_observation import BallObservation
from ball_trajectory import Ball, simulate_n_balls

class Particle(Ball):
    def __init__(self, x: float, y: float, vx: float, vy: float):
        super().__init__(x, y, vx, vy)

    def to_list(self):
        return [self.x, self.y, self.vx, self.vy]


class ParticleFilter:
    def __init__(self, num_particles: int, noise_stddev: float,
                 area: float = 100.0, noise: float = 0.2, gravity: float = -9.81):

        self.num_particles    = num_particles
        self.area             = area
        self.noise            = noise
        self.gravity          = gravity
        self._sigma           = noise_stddev
        self.rng              = np.random.default_rng()
        self.particles_history = []
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

        # # 3. Ground clamp: particles cannot exist below y=0
        # grounded          = self.y < 0
        # self.y[grounded]  = 0.0
        # self.vy[grounded] = 0.0

        self._sync_particles()
        return self.particles

    # ------------------------------------------------------------------
    # LIKELIHOOD  –  how well does each particle match the observation?
    # ------------------------------------------------------------------
    def likelihood(self, observation):
        obs_x, obs_y = observation
        distances    = np.sqrt((self.x - obs_x)**2 + (self.y - obs_y)**2)

        likelihoods  = np.exp(-0.5 * (distances / self._sigma)**2)
        likelihoods += 1e-300          # prevent all-zero weights

        return likelihoods / likelihoods.sum()

    # ------------------------------------------------------------------
    # RESAMPLE  –  multinomial resampling (only when weight diversity is low)
    # ------------------------------------------------------------------
    def resample(self):
        neff = 1.0 / np.sum(np.square(self.weights))
       # if neff >= self.num_particles / 2:
        if neff >= self.num_particles * 0.9:
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
        self.particles_history.append(self.particles.copy())
        return self.estimate()


# ======================================================================
# Example usage
# ======================================================================
if __name__ == "__main__":
    launches = [
        {"initial_position": [10, 15], "speed": 50, "angle_degrees": 45}
    ]

    trajectories = simulate_n_balls(launches, dt=0.1)
    noise_stddev = 8.0
    observers = [
        BallObservation(traj, noise_stddev=noise_stddev, dropout_prob=0.2)
        for traj in trajectories
    ]

    observations_ball0 = observers[0].simulate_observations()

    # Single filter run — estimates used for both printing and plotting
    pf = ParticleFilter(num_particles=1000, noise_stddev=noise_stddev, area=100.0, noise=1.0)
    estimates = []
    for t, obs in observations_ball0:
        estimate = pf.step(dt=0.1, observation=obs)
        estimates.append(estimate)
        print(f"Time: {t:.2f} s, Observation: {obs}")
        print(f"Estimated State: x={estimate[0]:.2f}, y={estimate[1]:.2f}, "
              f"vx={estimate[2]:.2f}, vy={estimate[3]:.2f}\n")

    plotter = ParticleFilterTrajectoryPlot(
        trajectory=trajectories[0],
        observations=observations_ball0,
        estimates=estimates
    )
    plotter.visualize()
 
    from plotting import ParticleEvolutionAnimation

    anim = ParticleEvolutionAnimation(
        particle_filter=pf,
        trajectory=trajectories[0],
        estimates=estimates
    )
    anim.animate()
