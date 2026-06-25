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
    def __init__(self, trajectories, noise_stddev=3.0, dropout_prob=0.15):
        super().__init__(figsize=(12, 6))
        self.observers = [
            BallObservation(traj, noise_stddev, dropout_prob)
            for traj in trajectories
        ]

    def visualize(self):
        for i, obs in enumerate(self.observers):
            c = self.colors[i % len(self.colors)]
            data = obs.simulate_observations()

            # Extract X–Y trajectory
            x = [p[1] for p in obs.trajectory]
            y = [p[2] for p in obs.trajectory]

            # True trajectory
            self.ax.plot(x, y, color=c, linewidth=2, label=f"True {i+1}")

            # Observations
            obs_x = [o[1][0] for o in data if o[1] is not None]
            obs_y = [o[1][1] for o in data if o[1] is not None]
            self.ax.scatter(obs_x, obs_y, color="gray", marker="x", s=20,
                            label="Obs" if i == 0 else None)

            # Bubble-style dropout visualization
            regions = merge_dropout(data)
            traj_times = [p[0] for p in obs.trajectory]

            for j, (s, e) in enumerate(regions):
                mid = (s + e) / 2

                # Find closest trajectory point
                idx = np.argmin(np.abs(np.array(traj_times) - mid))
                x_mid = x[idx]
                y_mid = y[idx]

                # Bubble size proportional to dropout duration
                bubble_size = 600 * (e - s)

                # Bubble
                self.ax.scatter(
                    x_mid, y_mid,
                    s=1200,
                    color="red",
                    alpha=0.18,
                    edgecolor="none",
                    label="Dropout" if i == 0 and j == 0 else None
                )

                # Label
                self.ax.text(
                    x_mid, y_mid + 5,
                    f"DROPOUT\n{mid:.2f}s",
                    ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3",
                              fc="yellow", ec="black")
                )

        self.finish("Observations with Dropout Bubbles",
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
        x = [p[1] for p in self.traj]
        y = [p[2] for p in self.traj]
        self.ax.plot(x, y, color="blue", linewidth=2, label="True")

        ox = [o[1][0] for o in self.obs if o[1] is not None]
        oy = [o[1][1] for o in self.obs if o[1] is not None]
        self.ax.scatter(ox, oy, color="gray", marker="x", s=25, label="Obs")

        # Bubble dropout
        for j, (s, e) in enumerate(merge_dropout(self.obs)):
            mid = (s + e) / 2
            idx = np.argmin(np.abs(np.array([p[0] for p in self.traj]) - mid))
            x_mid = self.traj[idx][1]
            y_mid = self.traj[idx][2]

            self.ax.scatter(x_mid, y_mid, s=900, color="red", alpha=0.18,
                            edgecolor="none", label="Dropout" if j == 0 else None)

            self.ax.text(
                x_mid, y_mid + 5,
                f"DROPOUT\n{mid:.2f}s",
                ha="center", va="center",
                fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3",
                          fc="yellow", ec="black")
            )

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

            tx = [p[1] for p in traj]
            ty = [p[2] for p in traj]
            self.ax.plot(tx, ty, color=c, linewidth=2, label=f"True {i+1}")

            ex = [e[0] for e in est_seq]
            ey = [e[1] for e in est_seq]
            self.ax.plot(ex, ey, "--", color=light, linewidth=2, label=f"Est {i+1}")

            ox = [o[1][0] for o in obs_seq if o[1] is not None]
            oy = [o[1][1] for o in obs_seq if o[1] is not None]
            self.ax.scatter(ox, oy, color="gray", marker="x", s=15,
                            label="Obs" if i == 0 else None)

            # Bubble dropout
            for j, (s, e) in enumerate(merge_dropout(obs_seq)):
                mid = (s + e) / 2
                idx = np.argmin(np.abs(np.array([p[0] for p in traj]) - mid))
                x_mid = traj[idx][1]
                y_mid = traj[idx][2]

                self.ax.scatter(
                    x_mid, y_mid,
                    s=900, color="red", alpha=0.18, edgecolor="none",
                    label="Dropout" if i == 0 and j == 0 else None
                )

                self.ax.text(
                    x_mid, y_mid + 5,
                    f"DROPOUT\n{mid:.2f}s",
                    ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3",
                              fc="yellow", ec="black")
                )

        self.finish("N-Ball Trajectories (Bubble Dropout)", "X Position (m)", "Y Position (m)")
