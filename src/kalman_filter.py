#-------------------
# AUTHORS
# DAMIAN - 10012545
# HARSHA - 10012956
#-------------------

import numpy as np
from ball_trajectory import simulate_n_balls
from ball_observation import BallObservation
from plotting import ParticleFilterTrajectoryPlot


class BallKalmanFilter:

    def __init__(self, noise_stddev: float, dt: float = 0.1,
                 gravity: float = -9.81, process_noise: float = 1.0):
        self.dt = dt

        self.F = np.array([[1, 0, dt,  0],
                           [0, 1,  0, dt],
                           [0, 0,  1,  0],
                           [0, 0,  0,  1]], dtype=float)
        self.u = np.array([0.0, 0.0, 0.0, gravity * dt])

        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=float)

        q      = process_noise
        self.Q = np.diag([q, q, q * 0.5, q * 0.5])
        self.R = np.eye(2) * noise_stddev ** 2

        self.x     = np.zeros(4)
        self.P     = np.eye(4) * 1e4
        self._ready = False

    def step(self, obs) -> np.ndarray:
        if obs is not None:
            z = np.array(obs, dtype=float)
            if not self._ready:
                self.x      = np.array([z[0], z[1], 0.0, 0.0])
                self.P      = np.eye(4) * 1e4
                self._ready = True
                return self.x.copy()

        if self._ready:
            self.x    = self.F @ self.x + self.u
            self.x[1] = max(self.x[1], 0.0)
            self.P    = self.F @ self.P @ self.F.T + self.Q

        if obs is not None:
            z          = np.array(obs, dtype=float)
            S          = self.H @ self.P @ self.H.T + self.R
            K          = self.P @ self.H.T @ np.linalg.inv(S)
            self.x     = self.x + K @ (z - self.H @ self.x)
            self.x[1]  = max(self.x[1], 0.0)
            self.P     = (np.eye(4) - K @ self.H) @ self.P

        return self.x.copy()


# ======================================================================
# EXAMPLE USAGE
# ======================================================================

if __name__ == "__main__":

    launches = [
        {"initial_position": [10, 15], "speed": 50, "angle_degrees": 45}
    ]
    DT           = 0.1
    N_BALLS      = len(launches)
    NOISE_STDDEV = 3.0
    DROPOUT_PROB = 0.2

    trajectories  = simulate_n_balls(launches, dt=DT)
    obs_sequences = [BallObservation(traj, noise_stddev=NOISE_STDDEV, dropout_prob=DROPOUT_PROB).simulate_observations()
                     for traj in trajectories]

    filters                = [BallKalmanFilter(noise_stddev=NOISE_STDDEV, dt=DT, process_noise=NOISE_STDDEV / 2)
                               for _ in range(N_BALLS)]
    estimated_trajectories = {i: [] for i in range(N_BALLS)}

    _cw = 15
    fmt = lambda v: f"({v[0]:6.1f},{v[1]:6.1f})" if v is not None else f"{'N/A':^{_cw}}"
    print(f"{'TIME STEP':^10}||" + "".join(f"{'BALL '+str(b):^{_cw*2+3}}||" for b in range(N_BALLS)))
    print(f"{'':10}||" + "".join(f"{'Obs.':^{_cw}} | {'Est.':^{_cw}}||" for _ in range(N_BALLS)))

    for step_idx in range(max(len(s) for s in obs_sequences)):
        t        = next(obs_sequences[b][step_idx][0] for b in range(N_BALLS) if step_idx < len(obs_sequences[b]))
        obs_list = [obs_sequences[b][step_idx][1] if step_idx < len(obs_sequences[b]) else None for b in range(N_BALLS)]
        est      = [filters[b].step(obs_list[b]) for b in range(N_BALLS)]

        for b in range(N_BALLS):
            if step_idx < len(trajectories[b]):
                estimated_trajectories[b].append(est[b])

        print(f"t={t:5.2f}s  ||" + "".join(
            f"{fmt(obs_list[b])} | {fmt(est[b][:2] if step_idx < len(trajectories[b]) else None)}||"
            for b in range(N_BALLS)))

    for i, (traj, obs_seq) in enumerate(zip(trajectories, obs_sequences)):
        ParticleFilterTrajectoryPlot(trajectory=traj, observations=obs_seq, estimates=estimated_trajectories[i]).visualize()
