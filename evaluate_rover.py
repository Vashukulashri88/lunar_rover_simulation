"""
evaluate_rover.py
Evaluates and benchmarks the Autonomous Lunar Rover on the Nakshatra Hackathon 2026 Criteria:
1. Hazard Detection Rate (% of negative & positive obstacles avoided)
2. Navigation Efficiency (Trajectory Optimality Ratio)
3. Autonomy Level (100% Zero Human Intervention)

Compares:
- Baseline: Naive Reactive / Greedy Direct-Path Controller
- Trained PPO Autonomous Navigation Agent
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from stable_baselines3 import PPO

from lunar_rover_gym_env import LunarRoverGymEnv
from lunar_rover_physics import MAX_RAY_RANGE_M


class NaiveReactiveBaseline:
    """
    Standard rule-based controller: tries to drive straight to goal,
    turning only when rays detect an obstacle immediately in front.
    Lacks predictive negative-obstacle (crater) awareness and slope stability modeling.
    """
    def __init__(self):
        pass

    def predict(self, obs, deterministic=True):
        # Extract observations
        # rays are first 15 entries
        rays = obs[:15]
        # target heading errors are at index 21, 22
        sin_err = obs[21]
        
        # Center rays (indices 5 to 9); indices 0..6 are Right (-angle), indices 8..14 are Left (+angle)
        center_min = np.min(rays[5:10])
        center_max = np.max(rays[5:10])
        right_min = np.min(rays[1:6])
        left_min = np.min(rays[9:14])
        
        # Hazard detected if:
        # 1. Positive obstacle (boulder): ray strikes early (center_min < 0.14)
        # 2. Negative obstacle (crater): ground drops off / ray plunges deep (center_max > 0.28)
        is_hazard = (center_min < 0.14) or (center_max > 0.28)
        
        if is_hazard:
            if left_min > right_min:
                action = np.array([-0.65, 0.65], dtype=np.float32) # Turn left (vR > vL -> +omega)
            else:
                action = np.array([0.65, -0.65], dtype=np.float32) # Turn right (vL > vR -> -omega)
        else:
            # Steer towards goal
            turn = np.clip(sin_err * 1.4, -0.5, 0.5)
            left = 0.75 - turn
            right = 0.75 + turn
            action = np.array([left, right], dtype=np.float32)
            
        return action, None


def run_benchmark(env, agent, num_episodes=50, agent_name="Agent"):
    successes = 0
    crater_fails = 0
    boulder_fails = 0
    rollover_fails = 0
    timeouts = 0
    trajectory_ratios = []
    episode_times = []
    all_trajectories = []
    all_terrains = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=ep * 101)
        terminated = False
        truncated = False
        total_steps = 0
        path_length = 0.0
        prev_pos = (env.rover.x, env.rover.y)
        
        while not (terminated or truncated):
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, step_info = env.step(action)
            total_steps += 1
            curr_pos = (env.rover.x, env.rover.y)
            path_length += np.hypot(curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1])
            prev_pos = curr_pos

        straight_dist = float(np.hypot(env.target_pos[0] - env.start_pos[0], env.target_pos[1] - env.start_pos[1]))
        optimality_ratio = path_length / max(0.1, straight_dist)

        if step_info.get("success", False):
            successes += 1
            trajectory_ratios.append(optimality_ratio)
            episode_times.append(total_steps * env.dt)
        elif step_info.get("failure_crater", False):
            crater_fails += 1
        elif step_info.get("failure_boulder", False):
            boulder_fails += 1
        elif step_info.get("failure_rollover", False):
            rollover_fails += 1
        else:
            timeouts += 1

        if ep < 5:  # Store first 5 episodes for detailed trajectory plotting
            all_trajectories.append({
                "trajectory": list(env.trajectory),
                "start": env.start_pos.copy(),
                "goal": env.target_pos.copy(),
                "craters": list(env.terrain.craters),
                "boulders": list(env.terrain.boulders),
                "success": step_info.get("success", False),
                "terrain": env.terrain
            })

    # Metrics summary
    hazard_detection_rate = 100.0 * (num_episodes - crater_fails - boulder_fails - rollover_fails) / num_episodes
    success_rate = 100.0 * successes / num_episodes
    mean_optimality = np.mean(trajectory_ratios) if trajectory_ratios else 0.0
    mean_time = np.mean(episode_times) if episode_times else 0.0

    metrics = {
        "name": agent_name,
        "episodes": num_episodes,
        "success_rate": success_rate,
        "hazard_detection_rate": hazard_detection_rate,
        "crater_falls": crater_fails,
        "boulder_collisions": boulder_fails,
        "rollovers": rollover_fails,
        "timeouts": timeouts,
        "optimality_ratio": mean_optimality,
        "mean_time_s": mean_time,
        "trajectories": all_trajectories
    }
    return metrics


def plot_evaluation_results(baseline_metrics, ppo_metrics, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Bar Chart Comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor('#0f172a')
    for ax in axes:
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#475569')

    names = ["Baseline (Naive Reactive)", "Trained PPO Agent"]
    colors = ['#f43f5e', '#10b981']

    # Metric 1: Hazard Detection & Avoidance Rate (%)
    ax = axes[0]
    hazard_rates = [baseline_metrics["hazard_detection_rate"], ppo_metrics["hazard_detection_rate"]]
    bars = ax.bar(names, hazard_rates, color=colors, width=0.55)
    ax.set_ylabel("Hazard Avoidance Rate (%)")
    ax.set_title("Metric 1: Hazard Detection Rate\n(Craters, Boulders & Slopes)", fontsize=11, fontweight='bold')
    ax.set_ylim(0, 115)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', color='white', fontweight='bold')

    # Metric 2: Navigation Efficiency (Trajectory Optimality Ratio)
    ax = axes[1]
    opt_ratios = [baseline_metrics["optimality_ratio"], ppo_metrics["optimality_ratio"]]
    bars = ax.bar(names, opt_ratios, color=colors, width=0.55)
    ax.set_ylabel("Actual Path / Shortest Path Ratio")
    ax.set_title("Metric 2: Trajectory Optimality\n(Lower = Shorter Detour)", fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(2.5, max(opt_ratios) * 1.3))
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"{yval:.2f}x", ha='center', va='bottom', color='white', fontweight='bold')

    # Metric 3: Autonomy Level (Zero-Intervention Success Rate %)
    ax = axes[2]
    success_rates = [baseline_metrics["success_rate"], ppo_metrics["success_rate"]]
    bars = ax.bar(names, success_rates, color=colors, width=0.55)
    ax.set_ylabel("Zero-Intervention Success (%)")
    ax.set_title("Metric 3: Autonomy Level\n(Goal Reached, Zero Interventions)", fontsize=11, fontweight='bold')
    ax.set_ylim(0, 115)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', color='white', fontweight='bold')

    plt.tight_layout()
    chart_path = os.path.join(save_dir, "evaluation_metrics.png")
    plt.savefig(chart_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[OK] Evaluation metrics chart saved to {chart_path}")

    # 2. Arena 2D Topdown Trajectory Comparison Map
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
    fig.patch.set_facecolor('#0f172a')
    
    for idx, (metrics, title) in enumerate([(baseline_metrics, "Baseline (Naive Reactive)"), (ppo_metrics, "Trained PPO Autonomous Agent")]):
        ax = axes[idx]
        ax.set_facecolor('#0f172a')
        ax.set_title(f"{title}\nCourse Trajectories & Hazard Avoidance", fontsize=12, fontweight='bold', color='white')
        if not metrics["trajectories"]:
            continue

        sample_ep = metrics["trajectories"][0]
        terrain = sample_ep["terrain"]
        half_bound = terrain.arena_size / 2.0

        ax.set_xlim(-half_bound * 1.05, half_bound * 1.05)
        ax.set_ylim(-half_bound * 1.05, half_bound * 1.05)
        ax.set_aspect('equal')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#475569')

        # Draw Terrain Elevation Contours
        grid_x, grid_y = np.meshgrid(np.linspace(-half_bound, half_bound, 100), np.linspace(-half_bound, half_bound, 100))
        grid_z = np.zeros_like(grid_x)
        for i in range(100):
            for j in range(100):
                grid_z[i, j] = terrain.get_height(grid_x[i, j], grid_y[i, j])
        
        contour = ax.contourf(grid_x, grid_y, grid_z, levels=20, cmap='bone', alpha=0.45)
        
        # Draw Craters (negative obstacles)
        for c in sample_ep["craters"]:
            crater_circle = Circle((c["x"], c["y"]), c["r"], color='#38bdf8', alpha=0.35, ec='#0284c7', lw=2.5, ls='--')
            ax.add_patch(crater_circle)
            ax.text(c["x"], c["y"], f"Crater\n(-{c['d']*100:.0f}cm)", ha='center', va='center', color='#bae6fd', fontsize=8, fontweight='bold')

        # Draw Boulders (positive obstacles)
        for b in sample_ep["boulders"]:
            boulder_circle = Circle((b["x"], b["y"]), b["r"], color='#d97706', alpha=0.75, ec='#f59e0b', lw=2)
            ax.add_patch(boulder_circle)
            ax.text(b["x"], b["y"], f"Rock", ha='center', va='center', color='black', fontsize=7, fontweight='bold')

        # Draw Trajectories
        for t_info in metrics["trajectories"][:4]:
            traj = np.array(t_info["trajectory"])
            color = '#22c55e' if t_info["success"] else '#ef4444'
            ax.plot(traj[:, 0], traj[:, 1], color=color, lw=2.5, alpha=0.9, zorder=5)
            # Start and Goal markers
            ax.scatter(t_info["start"][0], t_info["start"][1], color='#3b82f6', s=100, marker='o', zorder=6, label='Start' if idx == 0 else "")
            ax.scatter(t_info["goal"][0], t_info["goal"][1], color='#eab308', s=130, marker='*', zorder=6, label='Goal' if idx == 0 else "")

        # Legend
        if idx == 0:
            ax.plot([], [], color='#22c55e', lw=2.5, label='Success Trajectory')
            ax.plot([], [], color='#ef4444', lw=2.5, label='Collision / Trapped')
            ax.plot([], [], color='#38bdf8', lw=2, ls='--', label='Crater (Negative Drop)')
            ax.plot([], [], color='#f59e0b', lw=2, label='Boulder (Positive)')
            legend = ax.legend(loc='lower left', facecolor='#1e293b', edgecolor='#475569')
            for text in legend.get_texts():
                text.set_color('white')

    plt.tight_layout()
    traj_path = os.path.join(save_dir, "navigation_trajectories.png")
    plt.savefig(traj_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[OK] Trajectory comparison map saved to {traj_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained Lunar Rover Policy")
    parser.add_argument("--model-path", type=str, default="models/lunar_rover_ppo.zip")
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()

    env = LunarRoverGymEnv(arena_size=4.0, num_craters=4, num_boulders=6)

    # 1. Baseline Evaluation
    print("\n--- Evaluating Baseline Naive Reactive Controller ---")
    baseline = NaiveReactiveBaseline()
    baseline_metrics = run_benchmark(env, baseline, num_episodes=args.episodes, agent_name="Baseline")
    print(f"Baseline Success Rate:        {baseline_metrics['success_rate']:.1f}%")
    print(f"Baseline Hazard Avoidance:    {baseline_metrics['hazard_detection_rate']:.1f}%")
    print(f"Baseline Crater Falls:        {baseline_metrics['crater_falls']}")
    print(f"Baseline Boulder Collisions:  {baseline_metrics['boulder_collisions']}")
    print(f"Baseline Trajectory Ratio:    {baseline_metrics['optimality_ratio']:.2f}x")

    # 2. Trained PPO Evaluation
    if os.path.exists(args.model_path):
        print(f"\n--- Loading and Evaluating Trained PPO Agent ({args.model_path}) ---")
        ppo_model = PPO.load(args.model_path)
        ppo_metrics = run_benchmark(env, ppo_model, num_episodes=args.episodes, agent_name="PPO Agent")
    else:
        print(f"\n[Warning] Model {args.model_path} not found. Running baseline comparison.")
        ppo_metrics = baseline_metrics

    print(f"\nPPO Agent Success Rate:       {ppo_metrics['success_rate']:.1f}%")
    print(f"PPO Agent Hazard Avoidance:   {ppo_metrics['hazard_detection_rate']:.1f}%")
    print(f"PPO Agent Crater Falls:       {ppo_metrics['crater_falls']}")
    print(f"PPO Agent Boulder Collisions: {ppo_metrics['boulder_collisions']}")
    print(f"PPO Agent Trajectory Ratio:   {ppo_metrics['optimality_ratio']:.2f}x")

    # Plot & Save comparison charts
    plot_evaluation_results(baseline_metrics, ppo_metrics)


if __name__ == "__main__":
    main()
