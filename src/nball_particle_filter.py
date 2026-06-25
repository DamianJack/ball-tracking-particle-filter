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
        # Positions — uniform over the search area, above ground
        self.state[:, :, 0] = self.rng.uniform(-self.area / 2, self.area / 2, (N, B)) 
        self.state[:, :, 1] = self.rng.uniform(0, self.area / 2, (N, B))              

        # Velocities — random launch speed and upward angle
        speed = self.rng.uniform(20, 80, (N, B))
        angle = self.rng.uniform(np.deg2rad(20), np.deg2rad(80), (N, B))
        self.state[:, :, 2] = speed * np.cos(angle)  
        self.state[:, :, 3] = speed * np.sin(angle) 

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

        below = self.state[:, :, 1] < 0
        self.state[:, :, 1][below] = 0.0
        self.state[:, :, 3][below] = 0.0

    def likelihood(self, observations: list) -> np.ndarray:

        obs = [o for o in observations if o is not None]
        m   = len(obs)

        if m == 0:
            return np.ones(self.num_particles) / self.num_particles

        sigma = self._sigma
        perms = list(permutations(range(self.n_balls), m))
        logL  = np.zeros(self.num_particles)

        for p in range(self.num_particles):
            balls = self.state[p, :, :2]  
            best  = -np.inf

            for perm in perms:
                # perm[j] = which ball is matched to observation j
                d2 = 0.0
                for j, ball_idx in enumerate(perm):
                    ox, oy = obs[j]
                    bx, by = balls[ball_idx]
                    d2 += (bx - ox) ** 2 + (by - oy) ** 2

                log_like = -0.5 * d2 / sigma ** 2 
                if log_like > best:
                    best = log_like

            logL[p] = best

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

        self.weights *= self.likelihood(observations)
        self.weights /= self.weights.sum()
        self.resample()

    def estimate_states(self) -> np.ndarray:

        per_slot    = np.einsum('i,ijk->jk', self.weights, self.state)  # (n_balls, 4) fallback

        all_states  = self.state.reshape(-1, 4)                          # (N*B, 4)
        all_weights = np.repeat(self.weights, self.n_balls)
        positions   = all_states[:, :2]

        eps         = self._sigma * 2.0
        min_samples = max(3, len(positions) // 50)
        labels      = DBSCAN(eps=eps, min_samples=min_samples).fit(positions).labels_

        clusters = []
        for label in sorted(set(labels) - {-1}):
            m = labels == label
            w = all_weights[m]
            if w.sum() > 0:
                clusters.append(np.average(all_states[m], weights=w, axis=0))

        if not clusters:
            return per_slot

        estimated = per_slot.copy()
        for i, cluster in enumerate(clusters[:self.n_balls]):
            estimated[i] = cluster

        return estimated

    def step(self, dt: float, observations: list) -> np.ndarray:
        self.predict(dt)
        self.update(observations)
        return self.estimate_states()

# ======================================================================
# EXAMPLE USAGE
# ======================================================================

def assign_estimates(est: np.ndarray, true_positions: list) -> np.ndarray:

    assigned = np.empty_like(est)
    used = set()
    for b, tp in enumerate(true_positions):
        tp   = np.array(tp)
        best = min((j for j in range(len(est)) if j not in used),
                    key=lambda j: np.linalg.norm(est[j, :2] - tp))
        assigned[b] = est[best]
        used.add(best)
    return assigned

if __name__ == "__main__":

    launches     = [{"initial_position": [0,  0], "speed": 70, "angle_degrees": 35},
                    {"initial_position": [10, 0], "speed": 35, "angle_degrees": 65},
                    {"initial_position": [20, 0], "speed": 55, "angle_degrees": 50}]
    DT           = 0.1
    N_BALLS      = len(launches)
    NOISE_STDDEV = 3.0
    DROPOUT_PROB = 0.02

    trajectories  = simulate_n_balls(launches, dt=DT)
    observers     = [BallObservation(traj, noise_stddev=NOISE_STDDEV, dropout_prob=DROPOUT_PROB)
                     for traj in trajectories]
    obs_sequences = [obs.simulate_observations() for obs in observers]

    pf            = NBallParticleFilter(num_particles=5000, n_balls=N_BALLS, noise_stddev=NOISE_STDDEV, area=100.0, noise=NOISE_STDDEV/2)

    estimated_trajectories = {i: [] for i in range(N_BALLS)}
    n_steps                = max(len(seq) for seq in obs_sequences)

    _cw = 15  # width of each obs/est value cell
    print("=" * (10 + 2 + (N_BALLS * (_cw * 2 + 3 + 2))))
    print(f"{'TIME STEP':^10}||" + "".join(f"{'BALL '+str(b):^{_cw*2+3}}||" for b in range(N_BALLS)))
    print("=" * (10 + 2 + (N_BALLS * (_cw * 2 + 3 + 2))))
    print(f"{'':10}||" + "".join(f"{'Obs.':^{_cw}} | {'Est.':^{_cw}}||" for b in range(N_BALLS)))
    print("-" * (10 + 2 + (N_BALLS * (_cw * 2 + 3 + 2))))

    fmt = lambda v: f"({v[0]:6.1f},{v[1]:6.1f})" if v is not None else f"{'N/A':^{_cw}}"

    for step_idx in range(n_steps):
        t        = next(obs_sequences[b][step_idx][0] for b in range(N_BALLS) if step_idx < len(obs_sequences[b]))
        obs_list = [obs_sequences[b][step_idx][1] if step_idx < len(obs_sequences[b]) else None for b in range(N_BALLS)]
        true_pos = [trajectories[b][min(step_idx, len(trajectories[b])-1)][1:3] for b in range(N_BALLS)]
        est      = assign_estimates(pf.step(dt=DT, observations=obs_list), true_pos)

        for b in range(N_BALLS):
            if step_idx < len(trajectories[b]):
                estimated_trajectories[b].append(est[b])

        print(f"t={t:5.2f}s  ||" + "".join(
            f"{fmt(obs_list[b])} | {fmt(est[b, :2] if step_idx < len(trajectories[b]) else None)}||"
            for b in range(N_BALLS)))

    # Plot each ball
    for i, (traj, obs_seq) in enumerate(zip(trajectories, obs_sequences)):
        ParticleFilterTrajectoryPlot(trajectory=traj, observations=obs_seq, estimates=estimated_trajectories[i]).visualize()
