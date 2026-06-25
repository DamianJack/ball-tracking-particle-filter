import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from ball_trajectory import simulate_n_balls
from ball_observation import BallObservation


# ============================================================
# BASE CLASS
# ============================================================
class BasePlot:
    def __init__(self, figsize=(12, 6)):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        # LIST, not dict → avoids KeyError
        self.colors = ["red", "blue", "green", "orange", "purple", "brown"]

    def finish(self, title, xlabel, ylabel):
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.grid(True, alpha=0.4)
        self.ax.legend()
        plt.tight_layout()
        plt.show()


# ============================================================
# MERGE DROPOUT REGIONS
# ============================================================
def merge_dropout(obs):
    regions = []
    active = False
    start = None

    for t, o in obs:
        if o is None and not active:
            active = True
            start = t
        elif o is not None and active:
            regions.append((start, t))
            active = False

    if active:
        regions.append((start, obs[-1][0]))

    return regions


# ============================================================
# 1) BallTrajectoryPlot  → N balls
# ============================================================
class BallTrajectoryPlot(BasePlot):
    def __init__(self, launches, dt=0.1):
        super().__init__(figsize=(10, 6))
        self.trajectories = simulate_n_balls(launches, dt)

    def visualize(self):
        for i, traj in enumerate(self.trajectories):
            c = self.colors[i % len(self.colors)]
            x = [p[1] for p in traj]
            y = [p[2] for p in traj]
            self.ax.plot(x, y, color=c, linewidth=2, label=f"Ball {i+1}")

        self.finish("Ball Trajectories", "X Position (m)", "Y Position (m)")


# ============================================================
# 2) BallObservationPlot  → N balls
# ============================================================
class BallObservationPlot(BasePlot):
    def __init__(self, trajectories, noise_stddev=3.0, dropout_prob=0.05):
        super().__init__(figsize=(12, 6))
        self.observers = [
            BallObservation(traj, noise_stddev, dropout_prob)
            for traj in trajectories
        ]

    def visualize(self):
        for i, obs in enumerate(self.observers):
            c = self.colors[i % len(self.colors)]
            data = obs.simulate_observations()

            t = [p[0] for p in obs.trajectory]
            true_y = [p[2] for p in obs.trajectory]
            obs_y = [o[1][1] if o[1] is not None else np.nan for o in data]

            self.ax.plot(t, true_y, color=c, linewidth=2, label=f"True {i+1}")
            self.ax.scatter(t, obs_y, color="cyan", s=20, alpha=0.6,
                        label="Obs" if i == 0 else None)

            # Merge consecutive dropouts
            regions = []
            active = False
            start = None
            for tt, o in data:
                if o is None and not active:
                    active = True
                    start = tt
                elif o is not None and active:
                    regions.append((start, tt))
                    active = False
            if active:
                regions.append((start, data[-1][0]))

            # Plot merged dropout regions (single legend entry)
            for j, (s, e) in enumerate(regions):
                self.ax.axvspan(s, e, color="red", alpha=0.15,
                            label="Dropout" if i == 0 and j == 0 else None)

        self.finish("True vs Observed (with Dropout)", "Time (s)", "Y Position (m)")


# ============================================================
# 3) ParticleFilterTrajectoryPlot  → 1 ball
# ============================================================
class ParticleFilterTrajectoryPlot(BasePlot):
    def __init__(self, trajectory, observations, estimates):
        super().__init__(figsize=(10, 6))
        self.traj = trajectory
        self.obs = observations
        self.est = estimates

    def visualize(self):
        x = [p[1] for p in self.traj]
        y = [p[2] for p in self.traj]
        self.ax.plot(x, y, color="blue", linewidth=2, label="True")

        ox = [o[1][0] for o in self.obs if o[1] is not None]
        oy = [o[1][1] for o in self.obs if o[1] is not None]
        self.ax.scatter(ox, oy, color="green", s=25, label="Obs")

        for s, e in merge_dropout(self.obs):
            self.ax.axvspan(s, e, color="red", alpha=0.15, label="Dropout")

        ex = [e[0] for e in self.est]
        ey = [e[1] for e in self.est]
        self.ax.plot(ex, ey, "--", color="lightcoral", linewidth=2, label="Estimate")

        self.finish("Particle Filter Tracking", "X Position (m)", "Y Position (m)")


# ============================================================
# 4) nBallTrajectoryPlot  → N balls (True + Est + Obs)
# ============================================================
class nBallTrajectoryPlot(BasePlot):
    def __init__(self, trajectories, observations, estimates):
        super().__init__(figsize=(12, 6))
        self.traj = trajectories
        self.obs = observations
        self.est = estimates

    def visualize(self):
        for i, (traj, obs_seq, est_seq) in enumerate(zip(self.traj, self.obs, self.est)):
            c = self.colors[i % len(self.colors)]
            light = mcolors.to_rgba(c, alpha=0.45)

            tx = [p[1] for p in traj]
            ty = [p[2] for p in traj]
            self.ax.plot(tx, ty, color=c, linewidth=2, label=f"True {i+1}")

            ex = [e[0] for e in est_seq]
            ey = [e[1] for e in est_seq]
            self.ax.plot(ex, ey, "--", color=light, linewidth=2, label=f"Est {i+1}")

            ox = [o[1][0] for o in obs_seq if o[1] is not None]
            oy = [o[1][1] for o in obs_seq if o[1] is not None]
            self.ax.scatter(ox, oy, color="cyan", s=12, alpha=0.6,
                            label="Obs" if i == 0 else None)

            for s, e in merge_dropout(obs_seq):
                self.ax.axvspan(s, e, color="red", alpha=0.15,
                                label="Dropout" if i == 0 else None)

        self.finish("N-Ball Trajectories (True / Est / Obs)", "X Position (m)", "Y Position (m)")
