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
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.grid(True, alpha=0.4)
        self.ax.legend()
        plt.tight_layout()
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
        plt.close('all')
        fig, ax = plt.subplots(figsize=(10, 6))

        traj_x = [p[1] for p in self.trajectory]
        traj_y = [p[2] for p in self.trajectory]
        ax.plot(traj_x, traj_y, color="blue", linewidth=2, label="True Trajectory")

        obs_x = [o[1][0] for o in self.observations if o[1] is not None]
        obs_y = [o[1][1] for o in self.observations if o[1] is not None]
        ax.scatter(obs_x, obs_y, color="green", s=25, label="Observations")

        # Dropout visualization
        dropout_times = [o[0] for o in self.observations if o[1] is None]
        for t in dropout_times:
            ax.axvspan(t - 0.05, t + 0.05, color="red", alpha=0.2)
            ax.text(t, max(traj_y) * 0.9, f"DROPOUT\n{t:.2f}s", ha="center", va="center",
                fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black"))

        # Add one legend entry for dropout
        if dropout_times:
            ax.axvspan(dropout_times[0] - 0.05, dropout_times[0] + 0.05,
                   color="red", alpha=0.2, label="Dropout Period")

        est_x = [e[0] for e in self.estimates]
        est_y = [e[1] for e in self.estimates]
        ax.plot(est_x, est_y, color="lightcoral", linestyle="--", linewidth=2,
            label="Particle Filter Estimate")

        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_title("Particle Filter Tracking of Ball Trajectory")
        ax.legend(loc="lower center", frameon=True)
        ax.grid(True, alpha=0.4, linestyle="--")

    plt.show()

# ============================================================
# CLASS: ParticleEvolutionAnimation
# ============================================================
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class ParticleEvolutionAnimation:
    def __init__(self, particle_filter, trajectory, estimates):
        self.pf = particle_filter
        self.trajectory = trajectory
        self.estimates = estimates

    def animate(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim(0, max(p[1] for p in self.trajectory) + 10)
        ax.set_ylim(0, max(p[2] for p in self.trajectory) + 10)
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_title("Particle Evolution Over Time")

        # True and estimated trajectories
        traj_x = [p[1] for p in self.trajectory]
        traj_y = [p[2] for p in self.trajectory]
        est_x = [e[0] for e in self.estimates]
        est_y = [e[1] for e in self.estimates]

        ax.plot(traj_x, traj_y, color="blue", linewidth=2, label="True Trajectory")
        ax.plot(est_x, est_y, color="lightgreen", linestyle="--", linewidth=2.5, label="PF Estimate")
        ax.legend(loc="upper right")

        # Particle scatter
        scat = ax.scatter([], [], color="gray", alpha=0.3, s=10)

        # Update function for animation
        def update(frame):
            particles = self.pf.particles_history[frame]
            scat.set_offsets(particles[:, :2])
            return scat,

        ani = animation.FuncAnimation(
            fig, update, frames=len(self.pf.particles_history),
            interval=100, blit=True, repeat=False
        )

        plt.tight_layout()
        plt.show()

# ============================================================
# CLASS: nBallTrajectoryPlot
# ============================================================
class nBallTrajectoryPlot(BasePlot):
    def __init__(self, trajectories, observations, estimates):
        super().__init__(figsize=(10, 6))
        self.trajectories = trajectories
        self.observations = observations
        self.estimates = estimates

    def visualize(self):
        true_colors = ["red", "blue", "green", "orange", "purple"]

        for i, (traj, obs_seq, est_seq) in enumerate(
            zip(self.trajectories, self.observations, self.estimates)
        ):
            # -------------------------
            # TRUE TRAJECTORY (solid)
            # -------------------------
            true_x = [p[1] for p in traj]
            true_y = [p[2] for p in traj]

            base_color = true_colors[i % len(true_colors)]
            self.ax.plot(
                true_x, true_y,
                color=base_color,
                linewidth=2,
                label=f"True Ball {i+1}"
            )

            # ----------------------------------------------------
            # ESTIMATED TRAJECTORY (lighter dotted version)
            # ----------------------------------------------------
            est_x = [e[0] for e in est_seq]
            est_y = [e[1] for e in est_seq]

            # lighter version of the same color
            light_color = mcolors.to_rgba(base_color, alpha=0.45)

            self.ax.plot(
                est_x, est_y,
                linestyle="--",
                color=light_color,
                linewidth=2,
                label=f"Est Ball {i+1}"
            )

            # -------------------------
            # OBSERVATIONS (cyan dots)
            # -------------------------
            obs_x = [o[1][0] for o in obs_seq if o[1] is not None]
            obs_y = [o[1][1] for o in obs_seq if o[1] is not None]

            self.ax.scatter(
                obs_x, obs_y,
                color="cyan",
                s=12,
                alpha=0.5,
                label="Observations" if i == 0 else None
            )

        self.finalize(
            title="N-Ball Trajectories: True vs Estimated",
            xlabel="X Position (m)",
            ylabel="Y Position (m)"
        )