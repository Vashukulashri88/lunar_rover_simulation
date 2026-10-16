"""
train_ppo.py
Trains an autonomous lunar rover navigation policy using Proximal Policy Optimization (PPO).
The agent learns to:
1. Detect and avoid negative obstacles (craters and drop-offs)
2. Detect and navigate around positive obstacles (boulders and rocks)
3. Navigate slopes while maintaining pitch/roll stability
4. Drive efficiently toward mission goal destinations with zero human intervention.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from lunar_rover_gym_env import LunarRoverGymEnv


class TrainingMetricsCallback(BaseCallback):
    """Logs episode successes, hazard collisions, and rewards during training."""
    def __init__(self, check_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.successes = []
        self.crater_falls = []
        self.boulder_hits = []
        self.rollovers = []

    def _on_step(self) -> bool:
        # Collect episode info from vectorized env infos
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
            if info.get("success", False):
                self.successes.append(1)
            elif any([info.get("failure_crater", False), info.get("failure_boulder", False), info.get("failure_rollover", False)]):
                self.successes.append(0)
                
            if info.get("failure_crater", False):
                self.crater_falls.append(1)
            if info.get("failure_boulder", False):
                self.boulder_hits.append(1)
            if info.get("failure_rollover", False):
                self.rollovers.append(1)
        return True


def make_env(arena_size=10.0, num_craters=12, num_boulders=18, max_steps=800):
    def _init():
        return LunarRoverGymEnv(
            arena_size=arena_size,
            num_craters=num_craters,
            num_boulders=num_boulders,
            max_steps=max_steps
        )
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train PPO policy for Nakshatra Autonomous Lunar Rover")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--arena-size", type=float, default=10.0, help="Arena size in meters (10.0 for 10m x 10m)")
    parser.add_argument("--num-craters", type=int, default=12, help="Number of procedural craters")
    parser.add_argument("--num-boulders", type=int, default=18, help="Number of procedural boulders")
    parser.add_argument("--max-steps", type=int, default=800, help="Maximum steps per episode")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--save-path", type=str, default="models/lunar_rover_ppo", help="Path to save trained model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print(f"=== Nakshatra Aerospace 2026: Lunar Rover PPO Training ===")
    print(f"Arena Size:      {args.arena_size}m x {args.arena_size}m ({args.arena_size**2:.0f} m²)")
    print(f"Hazards:         {args.num_craters} craters, {args.num_boulders} boulders")
    print(f"Episode Steps:   {args.max_steps} max steps")
    print(f"Total Timesteps: {args.timesteps:,}")
    print(f"Parallel Envs:   {args.num_envs}")
    print(f"Target Save:     {args.save_path}.zip")

    # Vectorized training environments
    env_fns = [
        make_env(
            arena_size=args.arena_size,
            num_craters=args.num_craters,
            num_boulders=args.num_boulders,
            max_steps=args.max_steps
        ) for _ in range(args.num_envs)
    ]
    vec_env = DummyVecEnv(env_fns)

    # Evaluation environment
    eval_env = DummyVecEnv([
        make_env(
            arena_size=args.arena_size,
            num_craters=args.num_craters,
            num_boulders=args.num_boulders,
            max_steps=args.max_steps
        )
    ])

    # Callback
    metrics_cb = TrainingMetricsCallback(check_freq=1000)

    # PPO Policy Setup
    # Custom 2-layer MLP [128, 128] optimized for low-latency onboard inference (< 0.2ms)
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128])
    )

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=8,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=args.seed
    )

    print("\nStarting PPO optimization...")
    model.learn(total_timesteps=args.timesteps, callback=metrics_cb)

    # Save trained policy
    model.save(args.save_path)
    print(f"\n[OK] Trained model successfully saved to: {args.save_path}.zip")

    # Export policy weights summary
    param_count = sum(p.numel() for p in model.policy.parameters())
    print(f"Policy Network Parameter Count: {param_count:,} weights (Ultra-lightweight for Edge MCU/NPU)")

    # Automatically export edge policy weights for physical rover
    print("\nExporting zero-dependency edge weights for onboard deployment...")
    try:
        from export_edge_policy import export_policy_numpy
        export_policy_numpy(
            model_path=f"{args.save_path}.zip",
            output_py="onboard_navigator.py",
            output_json="models/policy_weights.json"
        )
        print("[OK] Onboard Edge Navigator updated.")
    except Exception as e:
        print(f"[Warning] Edge policy export failed: {e}")


if __name__ == "__main__":
    main()

