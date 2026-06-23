import matplotlib.pyplot as plt
import numpy as np
from ball_trajectory import simulate_n_balls
from ball_observation import BallObservation


# ============================================================
# BASE CLASS FOR ALL PLOTS
# ============================================================
class BasePlot:
    def __init__(self, figsize=(12, 6)):
        self.fig, self.ax = plt.subplots(figsize=figsize)

        # Shared color palette
        self.colors = {
            "true": "blue",
            "obs": "green",
            "dropout": "red"
        }

    def finalize(self, title, xlabel, ylabel):
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_title(title)
        self.ax.legend(loc="upper right")
        self.ax.grid(True)
        plt.show()


# ============================================================
# CLASS: BallTrajectoryPlot
# ============================================================
class BallTrajectoryPlot(BasePlot):
    def __init__(self, launches, dt=0.1):
        super().__init__(figsize=(10, 6))
        self.launches = launches
        self.dt = dt
        self.trajectories = simulate_n_balls(launches, dt=dt)

    def visualize(self):
        colors = ["blue", "green", "red", "purple", "orange"]

        for i, traj in enumerate(self.trajectories):
            x_vals = [p[1] for p in traj]
            y_vals = [p[2] for p in traj]

            self.ax.plot(
                x_vals,
                y_vals,
                color=colors[i % len(colors)],
                linewidth=2,
                label=f"Ball {i+1} Trajectory"
            )

        self.finalize(
            title="Ball Trajectories",
            xlabel="X Position (m)",
            ylabel="Y Position (m)"
        )


# ============================================================
# CLASS: BallObservationPlot
# ============================================================
class BallObservationPlot(BasePlot):
    def __init__(self, trajectories, noise_stddev=3.0, dropout_prob=0.05):
        super().__init__(figsize=(12, 6))
        np.random.seed(42)

        self.trajectories = trajectories
        self.noise_stddev = noise_stddev
        self.dropout_prob = dropout_prob

        self.observers = [
            BallObservation(traj, noise_stddev, dropout_prob)
            for traj in trajectories
        ]

    def visualize(self):
        for i, obs in enumerate(self.observers):
            observations = obs.simulate_observations()

            times = [p[0] for p in obs.trajectory]
            true_y = [p[2] for p in obs.trajectory]
            obs_y = [o[1][1] if o[1] is not None else np.nan for o in observations]

            # True trajectory
            self.ax.plot(
                times,
                true_y,
                color=self.colors["true"],
                linewidth=2,
                label=f"Ball {i+1} - True Y"
            )

            # Observations
            self.ax.scatter(
                times,
                obs_y,
                color=self.colors["obs"],
                s=25,
                label=f"Ball {i+1} - Observations"
            )

            # Dropout shading
            dropout_times = [t for t, o in observations if o is None]

            for j, t in enumerate(dropout_times):
                if j % 5 == 0:  # reduce clutter
                    self.ax.axvspan(
                        t - 0.05,
                        t + 0.05,
                        color=self.colors["dropout"],
                        alpha=0.2,
                        label="Dropout Period" if i == 0 and j == 0 else ""
                    )

                    self.ax.text(
                        t,
                        max(true_y) * 0.9,
                        f"DROPOUT\n{t:.2f}s",
                        color="black",
                        fontsize=8,
                        fontweight="bold",
                        ha="center",
                        va="center",
                        bbox=dict(
                            facecolor="yellow",
                            edgecolor="black",
                            boxstyle="round,pad=0.3"
                        )
                    )

        self.finalize(
            title="True vs Observed Trajectories with Dropout Periods",
            xlabel="Time (s)",
            ylabel="Y Position (m)"
        )

# ============================================================
# CLASS: ParticleFilterTrajectoryPlot
# ============================================================

class ParticleFilterTrajectoryPlot:
    def __init__(self, trajectory, observations, estimates):
        self.trajectory = trajectory
        self.observations = observations
        self.estimates = estimates  # pre-computed [(x, y), ...] from the single filter run

    def visualize(self):
        fig, ax = plt.subplots(figsize=(10, 6))

        # True trajectory
        traj_x = [p[1] for p in self.trajectory]
        traj_y = [p[2] for p in self.trajectory]
        ax.plot(traj_x, traj_y, color="blue", linewidth=2, label="True Trajectory")

        # Observations
        obs_x = [o[1][0] for o in self.observations if o[1] is not None]
        obs_y = [o[1][1] for o in self.observations if o[1] is not None]
        ax.scatter(obs_x, obs_y, color="red", s=25, label="Observations")

        # Pre-computed estimates — same values that were printed
        est_x = [e[0] for e in self.estimates]
        est_y = [e[1] for e in self.estimates]
        ax.plot(est_x, est_y, color="green", linewidth=2, label="Particle Filter Estimate")

        # Labels, legend, and grid
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_title("Particle Filter Tracking of Ball Trajectory")
        ax.legend(loc="lower center", frameon=True)
        ax.grid(True, alpha=0.4, linestyle="--")

        plt.show()
