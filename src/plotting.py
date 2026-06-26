import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from ball_trajectory import simulate_n_balls
from ball_observation import BallObservation


# ============================================================
# BASE CLASS
# ============================================================
OBS_COLOR     = "#9E9E9E"   # medium gray for all observations
DROPOUT_COLOR = "#DB2777"   # hot pink — shared by all dropout markers


class BasePlot:
    def __init__(self, figsize=(12, 6)):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.colors = ["#2563EB", "#EA580C", "#16A34A", "#9333EA", "#0891B2", "#CA8A04"]

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
def merge_dropout(obs, min_duration=0.15):
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

    # Filter out tiny dropout blips
    return [(s, e) for s, e in regions if e - s >= min_duration]


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

            # Landing marker
            self.ax.scatter(x[-1], y[-1], s=120, color=c, edgecolor="black")

        self.finish("Ball Trajectories", "X Position (m)", "Y Position (m)")


# ============================================================
# 2) BallObservationPlot  → N balls + bubble dropout
# ============================================================
class BallObservationPlot(BasePlot):
    def __init__(self, trajectories, obs_sequences):
        super().__init__(figsize=(12, 6))
        self.trajectories = trajectories
        self.obs_sequences = obs_sequences

    def visualize(self):
        for i, (traj, data) in enumerate(zip(self.trajectories, self.obs_sequences)):
            c = self.colors[i % len(self.colors)]

            x = [p[1] for p in traj]
            y = [p[2] for p in traj]
            traj_times = np.array([p[0] for p in traj])

            self.ax.plot(x, y, color=c, linewidth=2, label=f"True {i+1}")

            obs_x = [o[1][0] for o in data if o[1] is not None]
            obs_y = [o[1][1] for o in data if o[1] is not None]
            self.ax.scatter(obs_x, obs_y, color=OBS_COLOR, marker="x", s=20,
                            label="Obs" if i == 0 else None)

            drop_x = [x[np.argmin(np.abs(traj_times - t))] for t, o in data if o is None]
            drop_y = [y[np.argmin(np.abs(traj_times - t))] for t, o in data if o is None]
            if drop_x:
                self.ax.scatter(drop_x, drop_y, color=DROPOUT_COLOR, marker="x", s=100,
                                linewidths=2.5, label="Dropout" if i == 0 else None)

        self.finish("Observations with Dropout Markers",
                    "X Position (m)", "Y Position (m)")


# ============================================================
# 3) ParticleFilterTrajectoryPlot  → 1 ball + bubble dropout
# ============================================================
class ParticleFilterTrajectoryPlot(BasePlot):
    def __init__(self, trajectory, observations, estimates):
        super().__init__(figsize=(10, 6))
        self.traj = trajectory
        self.obs = observations
        self.est = estimates

    def visualize(self):
        tt = np.array([p[0] for p in self.traj])
        x  = [p[1] for p in self.traj]
        y  = [p[2] for p in self.traj]
        self.ax.plot(x, y, color=self.colors[0], linewidth=2, label="True")

        ox = [o[1][0] for o in self.obs if o[1] is not None]
        oy = [o[1][1] for o in self.obs if o[1] is not None]
        self.ax.scatter(ox, oy, color=OBS_COLOR, marker="x", s=25, label="Obs")

        drop_x = [x[np.argmin(np.abs(tt - t))] for t, o in self.obs if o is None]
        drop_y = [y[np.argmin(np.abs(tt - t))] for t, o in self.obs if o is None]
        if drop_x:
            self.ax.scatter(drop_x, drop_y, color=DROPOUT_COLOR, marker="x", s=100,
                            linewidths=2.5, label="Dropout")

        ex = [e[0] for e in self.est]
        ey = [e[1] for e in self.est]
        self.ax.plot(ex, ey, "--", color="lightcoral", linewidth=2, label="Estimate")

        self.finish("Particle Filter Tracking", "X Position (m)", "Y Position (m)")


# ============================================================
# 4) nBallTrajectoryPlot  → N balls + bubble dropout
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

            tt = np.array([p[0] for p in traj])
            tx = [p[1] for p in traj]
            ty = [p[2] for p in traj]
            self.ax.plot(tx, ty, color=c, linewidth=2, label=f"True {i+1}")

            ex = [e[0] for e in est_seq]
            ey = [e[1] for e in est_seq]
            self.ax.plot(ex, ey, "--", color=light, linewidth=2, label=f"Est {i+1}")

            ox = [o[1][0] for o in obs_seq if o[1] is not None]
            oy = [o[1][1] for o in obs_seq if o[1] is not None]
            self.ax.scatter(ox, oy, color=OBS_COLOR, marker="x", s=15,
                            label="Obs" if i == 0 else None)

            drop_x = [tx[np.argmin(np.abs(tt - t))] for t, o in obs_seq if o is None]
            drop_y = [ty[np.argmin(np.abs(tt - t))] for t, o in obs_seq if o is None]
            if drop_x:
                self.ax.scatter(drop_x, drop_y, color=DROPOUT_COLOR, marker="x", s=100,
                                linewidths=2.5, label="Dropout" if i == 0 else None)

        self.finish("N-Ball Trajectories with Dropout Markers", "X Position (m)", "Y Position (m)")

# ============================================================
# 5) BallCountPlot  → shows number of balls over time
# ============================================================
class BallCountPlot(BasePlot):
    def __init__(self, time, estimated_balls, observed_balls):
        super().__init__(figsize=(10, 6))
        self.time = time
        self.estimated_balls = estimated_balls
        self.observed_balls = observed_balls

    def visualize(self):
        self.ax.plot(self.time, self.estimated_balls, 'b--', linewidth=2, label='Estimated Balls')
        self.ax.plot(self.time, self.observed_balls, 'c:', linewidth=2, label='Observations')
        self.finish("Ball Count Over Time", "Time (s)", "Number of Balls")

# ============================================================
# 6) TrackingErrorPlot → shows error vs dropout periods
# ============================================================
class TrackingErrorPlot(BasePlot):
    def __init__(self, time, errors, dropout_regions):
        super().__init__(figsize=(12, 6))
        self.time = time
        self.errors = errors
        self.dropout_regions = dropout_regions

    def visualize(self):
        # Dropout shading
        for j, (s, e) in enumerate(self.dropout_regions):
            self.ax.axvspan(s, e, color="red", alpha=0.15,
                            label="Dropout Window" if j == 0 else None)

        # Error curves
        for i, (ball_idx, err) in enumerate(self.errors.items()):
            c = self.colors[i % len(self.colors)]
            self.ax.plot(self.time[:len(err)], err, color=c, linewidth=2,
                         label=f"Ball {ball_idx} Error")

        self.finish("Tracking Error vs Sensor Dropout Periods",
                    "Time (s)", "Position Error (m)")

def compute_tracking_errors(trajectories, estimated_trajectories):
    errors = {}

    for b, traj in enumerate(trajectories):
        true_xy = [p[1:3] for p in traj]      # (x, y)
        est_xy  = estimated_trajectories[b]   # (x, y, vx, vy)

        n = min(len(true_xy), len(est_xy))
        err = []

        for i in range(n):
            tx, ty = true_xy[i]
            ex, ey = est_xy[i][:2]            # <-- FIX HERE
            err.append(np.linalg.norm([tx - ex, ty - ey]))

        errors[b] = err

    return errors



def extract_dropout_regions(obs_sequences):
    regions = []
    for seq in obs_sequences:
        for s, e in merge_dropout(seq):
            regions.append((s, e))
    return regions
