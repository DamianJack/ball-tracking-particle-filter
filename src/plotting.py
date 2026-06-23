import matplotlib.pyplot as plt
import numpy as np
from ball_trajectory import simulate_n_balls
from ball_observation import BallObservation
from particle_filter import ParticleFilter


# ============================================================
# CLASS: BallTrajectoryPlot
# ============================================================
class BallTrajectoryPlot:
    def __init__(self, launches, dt=0.1):
        self.launches = launches
        self.dt = dt
        self.trajectories = simulate_n_balls(launches, dt=dt)

    def visualize(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["blue", "green", "red", "purple", "orange"]

        for i, traj in enumerate(self.trajectories):
            x_vals = [p[1] for p in traj]
            y_vals = [p[2] for p in traj]

            ax.plot(
                x_vals,
                y_vals,
                color=colors[i % len(colors)],
                linewidth=2,
                label=f"Ball {i+1} Trajectory"
            )

        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_title("Ball Trajectories")
        ax.legend()
        ax.grid(True)
        plt.show()


# ============================================================
# CLASS: BallObservationPlot
# ============================================================
class BallObservationPlot:
    def __init__(self, trajectories, noise_stddev=3.0, dropout_prob=0.05):
        np.random.seed(42)
        self.trajectories = trajectories
        self.noise_stddev = noise_stddev
        self.dropout_prob = dropout_prob
        self.observers = [
            BallObservation(traj, noise_stddev, dropout_prob)
            for traj in trajectories
        ]

    def visualize(self):
        fig, ax = plt.subplots(figsize=(12, 6))

        line_color = "blue"
        dot_color = "green"
        dropout_color = "red"

        for i, obs in enumerate(self.observers):
            observations = obs.simulate_observations()

            times = [p[0] for p in obs.trajectory]
            true_y = [p[2] for p in obs.trajectory]
            obs_y = [o[1][1] if o[1] is not None else np.nan for o in observations]

            ax.plot(times, true_y, color=line_color, linewidth=2,
                    label=f"Ball {i+1} - True Y")

            ax.scatter(times, obs_y, color=dot_color, s=25,
                       label=f"Ball {i+1} - Observations")

            dropout_times = [t for t, o in observations if o is None]

            for j, t in enumerate(dropout_times):
                if j % 5 == 0:
                    ax.axvspan(t - 0.05, t + 0.05,
                               color=dropout_color, alpha=0.2,
                               label="Dropout Period" if i == 0 and j == 0 else "")

                    ax.text(t, max(true_y) * 0.9,
                            f"DROPOUT\n{t:.2f}s",
                            color="black", fontsize=8, fontweight="bold",
                            ha="center", va="center",
                            bbox=dict(facecolor="yellow",
                                      edgecolor="black",
                                      boxstyle="round,pad=0.3"))

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Y Position (m)")
        ax.set_title("True vs Observed Trajectories with Dropout Periods")
        ax.legend(loc="upper right")
        ax.grid(True)
        plt.show()


#============================================================
## class: ParticleEvolutionPlot
#============================================================

class ParticleEvolutionPlot:
    def __init__(self, trajectory, observations, particle_filter, dt=0.1):
        self.trajectory = trajectory
        self.observations = observations
        self.pf = particle_filter
        self.dt = dt

    def visualize(self):
        fig, ax = plt.subplots(figsize=(12, 6))

        # Colors
        true_color = "blue"
        obs_color = "green"
        est_color = "red"
        particle_color = "gray"
        dropout_color = "red"

        # True trajectory
        true_x = [p[1] for p in self.trajectory]
        true_y = [p[2] for p in self.trajectory]
        ax.plot(true_x, true_y, color=true_color, linewidth=2, label="True Trajectory")

        # Observations
        obs_x = [o[1][0] for o in self.observations if o[1] is not None]
        obs_y = [o[1][1] for o in self.observations if o[1] is not None]
        ax.scatter(obs_x, obs_y, color=obs_color, s=25, label="Observations")

        # Particle filter estimates
        est_x, est_y = [], []

        dropout_times = [t for t, o in self.observations if o is None]

        for j, (t, obs) in enumerate(self.observations):
            # Predict + update
            estimate = self.pf.step(self.dt, obs)
            est_x.append(estimate[0])
            est_y.append(estimate[1])

            # Plot particles (gray cloud)
            ax.scatter(self.pf.x, self.pf.y, color=particle_color, s=5, alpha=0.3)

            # Highlight dropout periods
            if obs is None:
                ax.axvspan(t - 0.05, t + 0.05, color=dropout_color, alpha=0.2,
                           label="Dropout Period" if j == 0 else "")
                ax.text(t, max(true_y)*0.9, f"DROPOUT\n{t:.2f}s",
                        color="black", fontsize=8, fontweight="bold",
                        ha="center", va="center",
                        bbox=dict(facecolor="yellow", edgecolor="black", boxstyle="round,pad=0.3"))

        # Plot PF estimate trajectory
        ax.plot(est_x, est_y, color=est_color, linewidth=2, label="PF Estimate")

        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_title("Particle Evolution with Noise and Dropout")
        ax.legend(loc="upper right")
        ax.grid(True)
        plt.show()

