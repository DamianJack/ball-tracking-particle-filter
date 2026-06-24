#-------------------
# AUTHORS
# DAMIAN - 10012545
# HARSHA - 10012956
#-------------------

import numpy as np
from itertools import permutations
from sklearn.cluster import DBSCAN

from ball_trajectory import simulate_n_balls
from ball_observation import BallObservation
from plotting import ParticleFilterTrajectoryPlot


class NBallParticleFilter:
    """
    Extension of the single-ball ParticleFilter to n balls tracked jointly.

    Single-ball state:  four arrays  self.x, self.y, self.vx, self.vy  each (N,)
    N-ball state:       one array    self.state                          (N, n_balls, 4)
                        columns:  [x, y, vx, vy]

    Every method maps directly from the single-ball version.  The only
    structural differences are:

      likelihood   — balls are indistinguishable, so we enumerate all
                     possible assignments of observations to balls using
                     permutations instead of comparing to a single ball.

      estimate_states — the posterior is multimodal (label-switching), so
                        a simple weighted mean lands between the balls.
                        DBSCAN clusters the particle cloud instead.
    """

    def __init__(self, num_particles: int, n_balls: int, noise_stddev: float,
                 area: float = 100.0, noise: float = 1.0, gravity: float = -9.81):
        self.num_particles = num_particles
        self.n_balls       = n_balls
        self._sigma        = noise_stddev
        self.area          = area
        self.noise         = noise
        self.gravity       = gravity
        self.rng           = np.random.default_rng()
        self.initialize_particles()

    def initialize_particles(self):
       
        N, B = self.num_particles, self.n_balls

        self.state = np.zeros((N, B, 4))

        print(f"Initializing {N} particles for {B} balls in area {self.area}x{self.area}")
        print(self.state.shape)
        # Positions — uniform over the search area, above ground
        self.state[:, :, 0] = self.rng.uniform(-self.area / 2, self.area / 2, (N, B)) 
        self.state[:, :, 1] = self.rng.uniform(0, self.area / 2, (N, B))              

        # Velocities — random launch speed and upward angle
        speed = self.rng.uniform(0, 60, (N, B))
        angle = self.rng.uniform(0, np.pi, (N, B))  # upward launches only
        self.state[:, :, 2] = speed * np.cos(angle)  # vx
        self.state[:, :, 3] = speed * np.sin(angle)  # vy

        # Uniform weights — all particles equally likely before first observation
        self.weights = np.ones(N) / N

    def predict(self, dt: float):
       
        self.state[:, :, 0] += self.state[:, :, 2] * dt  
        self.state[:, :, 1] += self.state[:, :, 3] * dt  
        self.state[:, :, 3] += self.gravity * dt          

        N, B = self.num_particles, self.n_balls
        self.state[:, :, 0] += self.rng.normal(0, self.noise,       (N, B))  
        self.state[:, :, 1] += self.rng.normal(0, self.noise,       (N, B))  
        self.state[:, :, 2] += self.rng.normal(0, self.noise * 0.5, (N, B))  
        self.state[:, :, 3] += self.rng.normal(0, self.noise * 0.5, (N, B))  

        # Ground clamp 
        below = self.state[:, :, 1] < 0
        self.state[:, :, 1][below] = 0.0
        self.state[:, :, 3][below] = 0.0

    def likelihood(self, observations: list) -> np.ndarray:

        obs = [o for o in observations if o is not None]
        m   = len(obs)

        # No valid observations this step — keep weights uniform (no info)
        if m == 0:
            return np.ones(self.num_particles) / self.num_particles

        sigma = self._sigma
        perms = list(permutations(range(self.n_balls), m))
        logL  = np.zeros(self.num_particles)

        for p in range(self.num_particles):
            balls = self.state[p, :, :2]   # (n_balls, 2) — [x, y] per ball
            best  = -np.inf

            for perm in perms:
                # perm[j] = which ball is matched to observation j
                d2 = 0.0
                for j, ball_idx in enumerate(perm):
                    ox, oy = obs[j]
                    bx, by = balls[ball_idx]
                    d2 += (bx - ox) ** 2 + (by - oy) ** 2

                log_like = -0.5 * d2 / sigma ** 2  # same Gaussian formula as single-ball
                if log_like > best:
                    best = log_like

            logL[p] = best

        # Same numerically stable normalization as single-ball
        logL -= np.max(logL)
        L     = np.exp(logL)
        L    += 1e-300
        return L / L.sum()

    def resample(self):
        
        neff = 1.0 / np.sum(np.square(self.weights))
        if neff >= self.num_particles / 2:
            return  

        indices      = self.rng.choice(self.num_particles, size=self.num_particles,
                                       replace=True, p=self.weights)
        self.state   = self.state[indices]  
        self.weights = np.ones(self.num_particles) / self.num_particles

    def update(self, observations: list):
        self.weights = self.likelihood(observations)
        self.resample()

    def estimate_states(self) -> np.ndarray:
       
        all_states  = self.state.reshape(-1, 4)               
        all_weights = np.repeat(self.weights, self.n_balls)  
        positions   = all_states[:, :2]                       

        eps         = self._sigma * 2.0
        min_samples = max(5, self.num_particles // 100)
        labels      = DBSCAN(eps=eps, min_samples=min_samples).fit(positions).labels_

        estimated = []
        for label in sorted(set(labels) - {-1}):   
            mask = labels == label
            w    = all_weights[mask]
            w    = w / w.sum()
            estimated.append(np.average(all_states[mask], weights=w, axis=0))

        if len(estimated) < self.n_balls:
            fallback = np.einsum('i,ijk->jk', self.weights, self.state)  
            while len(estimated) < self.n_balls:
                estimated.append(fallback[len(estimated)])

        return np.array(estimated[:self.n_balls])  

    def step(self, dt: float, observations: list) -> np.ndarray:
        self.predict(dt)
        self.update(observations)
        return self.estimate_states()

# ======================================================================
# EXAMPLE USAGE
# ======================================================================
if __name__ == "__main__":

    launches = [
        {"initial_position": [0, 0], "speed": 50, "angle_degrees": 45},
        {"initial_position": [0, 0], "speed": 60, "angle_degrees": 30},
    ]

    DT           = 0.1
    N_BALLS      = len(launches)
    NOISE_STDDEV = 5.0
    DROPOUT_PROB = 0.02
    trajectories = simulate_n_balls(launches, dt=DT)

    observers     = [BallObservation(traj, noise_stddev=NOISE_STDDEV, dropout_prob=DROPOUT_PROB)
                     for traj in trajectories]
    obs_sequences = [obs.simulate_observations() for obs in observers]

    pf = NBallParticleFilter(
        num_particles=5000,
        n_balls=N_BALLS,
        noise_stddev=NOISE_STDDEV,
        area=100.0,
        noise=1.0,
    )

    estimates = []
    for step_bundle in zip(*obs_sequences):
        t        = step_bundle[0][0]
        obs_list = [entry[1] for entry in step_bundle]

        est = pf.step(dt=DT, observations=obs_list)
        estimates.append(est)

        print(f"t={t:.2f}s  " +
              "  ".join(f"Ball{i}: x={est[i,0]:.1f} y={est[i,1]:.1f} obs={obs_list[i]}" for i in range(N_BALLS)))


    # Plot each ball
    for i, (traj, obs_seq) in enumerate(zip(trajectories, obs_sequences)):
        ParticleFilterTrajectoryPlot(
            trajectory=traj,
            observations=obs_seq,
            estimates=[e[i] for e in estimates],
        ).visualize()
