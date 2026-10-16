"""
lunar_rover_gym_env.py
Gymnasium Environment for Nakshatra Aerospace Hackathon 2026:
Autonomous Lunar Rover for Surface Exploration and Hazard Avoidance.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple, Optional

from lunar_rover_physics import (
    LunarTerrain,
    LunarRoverPhysics,
    MAX_LENGTH_M,
    MAX_WIDTH_M,
    ROLLOVER_ANGLE_RAD,
    MAX_RAY_RANGE_M,
    NUM_RAYS
)

class LunarRoverGymEnv(gym.Env):
    """
    Custom Gymnasium Environment for training autonomous lunar rover navigation.
    Directly aligns with Hackathon evaluation metrics:
    1. Hazard Detection Rate (Craters, Boulders, Steep Slopes)
    2. Navigation Efficiency (Direct & Smooth Trajectory vs Detour)
    3. Autonomy Level (Zero interventions)
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        arena_size: float = 10.0,
        num_craters: int = 12,
        num_boulders: int = 18,
        max_steps: int = 800,
        dt: float = 0.05,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        self.arena_size = arena_size
        self.num_craters = num_craters
        self.num_boulders = num_boulders
        self.max_steps = max_steps
        self.dt = dt
        self.render_mode = render_mode
        
        # 2 Continuous motor throttles: [Left Track, Right Track] in [-1.0, 1.0]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )
        
        # Observation vector:
        # 15 rays + pitch + roll + yaw_rate + linear_speed + slip_ratio + target_dist + sin_target_err + cos_target_err = 23 dims
        obs_dim = NUM_RAYS + 8
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        self.terrain = None
        self.rover = None
        self.target_pos = np.array([1.2, 1.2], dtype=np.float32)
        self.start_pos = np.array([-1.2, -1.2], dtype=np.float32)
        
        self.step_count = 0
        self.prev_dist_to_goal = 0.0
        self.trajectory = []
        self.last_action = np.zeros(2, dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        rays = self.rover.get_raycast_observations()
        imu = self.rover.get_imu_state()
        
        # Target relative calculations
        dx = self.target_pos[0] - self.rover.x
        dy = self.target_pos[1] - self.rover.y
        dist = np.hypot(dx, dy)
        max_arena_diag = self.arena_size * np.sqrt(2.0)
        norm_dist = np.clip(dist / max_arena_diag, 0.0, 1.0)
        
        target_heading = np.arctan2(dy, dx)
        heading_err = (target_heading - self.rover.yaw + np.pi) % (2 * np.pi) - np.pi
        sin_err = np.sin(heading_err)
        cos_err = np.cos(heading_err)
        
        obs = np.concatenate([
            rays,  # 15 dims
            np.array([
                self.rover.pitch,                   # rad
                self.rover.roll,                    # rad
                self.rover.omega,                   # rad/s
                imu["linear_speed_ms"],             # m/s
                self.rover.slip_ratio,              # [0, 1]
                norm_dist,                          # [0, 1]
                sin_err,                            # [-1, 1]
                cos_err                             # [-1, 1]
            ], dtype=np.float32)
        ]).astype(np.float32)
        
        return obs

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        # Procedurally generate new arena configuration
        self.terrain = LunarTerrain(
            arena_size=self.arena_size,
            num_craters=self.num_craters,
            num_boulders=self.num_boulders,
            seed=seed
        )
        self.rover = LunarRoverPhysics(self.terrain)
        
        # Spawn in a corner region with target across the arena (~2.5m - 3.2m away)
        side_span = (self.arena_size / 2.0) - 0.7
        angle = self.np_random.uniform(0, 2 * np.pi)
        dist_spawn = self.np_random.uniform(1.0, side_span)
        start_x = dist_spawn * np.cos(angle)
        start_y = dist_spawn * np.sin(angle)
        
        # Target on opposite side with slight random offset
        goal_x = -start_x * self.np_random.uniform(0.7, 1.0) + self.np_random.uniform(-0.3, 0.3)
        goal_y = -start_y * self.np_random.uniform(0.7, 1.0) + self.np_random.uniform(-0.3, 0.3)
        
        goal_x = float(np.clip(goal_x, -side_span, side_span))
        goal_y = float(np.clip(goal_y, -side_span, side_span))
            
        initial_yaw = float(np.arctan2(goal_y - start_y, goal_x - start_x))
        self.start_pos = np.array([start_x, start_y], dtype=np.float32)
        self.target_pos = np.array([goal_x, goal_y], dtype=np.float32)
        
        # Ensure neither start nor target spawns inside a crater or boulder
        self.terrain.ensure_safe_zones([(start_x, start_y), (goal_x, goal_y)], min_clearance=0.85)
        
        self.rover.reset((start_x, start_y, initial_yaw))
        self.prev_dist_to_goal = float(np.hypot(goal_x - start_x, goal_y - start_y))
        self.step_count = 0
        self.trajectory = [(self.rover.x, self.rover.y, self.rover.z)]
        self.last_action = np.zeros(2, dtype=np.float32)
        
        obs = self._get_obs()
        info = {
            "start_pos": self.start_pos,
            "target_pos": self.target_pos,
            "dist_to_goal": self.prev_dist_to_goal
        }
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1
        left_cmd, right_cmd = float(action[0]), float(action[1])
        
        # Step physics simulator
        self.rover.step(left_cmd, right_cmd, dt=self.dt)
        self.trajectory.append((self.rover.x, self.rover.y, self.rover.z))
        
        # Calculate distance to goal
        dist = float(np.hypot(self.target_pos[0] - self.rover.x, self.target_pos[1] - self.rover.y))
        progress = self.prev_dist_to_goal - dist
        self.prev_dist_to_goal = dist
        
        # Reward components
        reward = 0.0
        terminated = False
        truncated = False
        
        # 1. Navigation Efficiency Progress Reward (strong incentive to drive toward goal)
        reward += progress * 25.0
        
        # 2. Step cost (small penalty to encourage expeditious path)
        reward -= 0.02
        
        # 3. Action smoothness / Anti-zigzag & Anti-circling penalty
        steer_diff = abs(left_cmd - right_cmd)
        reward -= 0.015 * steer_diff
        if progress <= 0.001 and steer_diff > 0.6:
            # Penalize spinning/circling in place without making progress toward goal
            reward -= 0.08 * steer_diff
        
        # 4. Proactive Obstacle Warning Penalty (Nominal flat ray return is ~0.228):
        # Penalize both positive obstacles (boulder: ray < 0.16) and negative obstacles (crater drop-off: ray > 0.28)
        rays = self.rover.get_raycast_observations()
        min_center_ray = float(np.min(rays[5:10]))
        max_center_ray = float(np.max(rays[5:10]))
        if min_center_ray < 0.16:
            reward -= 0.35 * (0.16 - min_center_ray)
        if max_center_ray > 0.28:
            reward -= 0.45 * (max_center_ray - 0.28)
        
        # 5. Slope / Tilt penalty (prevent high roll/pitch instability)
        tilt_magnitude = np.hypot(self.rover.pitch, self.rover.roll)
        reward -= 1.0 * (tilt_magnitude**2)
        
        # Termination conditions based on Hackathon Judging criteria:
        info = {
            "success": False,
            "failure_crater": False,
            "failure_boulder": False,
            "failure_rollover": False,
            "failure_oob": False,
            "dist_to_goal": dist,
            "trajectory_len": len(self.trajectory)
        }
        
        # Check Success: Rover reached goal zone (within 20 cm)
        if dist < 0.22:
            reward += 120.0 + (self.max_steps - self.step_count) * 0.1
            terminated = True
            info["success"] = True
            
        # Check Failures:
        # Negative obstacle failure: Trapped or tipped into crater
        elif self.rover.is_trapped_in_crater:
            reward -= 120.0
            terminated = True
            info["failure_crater"] = True
            
        # Positive obstacle failure: Collided with boulder
        elif self.rover.has_collided_boulder:
            reward -= 80.0
            terminated = True
            info["failure_boulder"] = True
            
        # Rollover / Tip-over failure
        elif self.rover.has_rolled_over:
            reward -= 90.0
            terminated = True
            info["failure_rollover"] = True
            
        # Arena boundary exit
        elif self.rover.is_out_of_bounds:
            reward -= 50.0
            terminated = True
            info["failure_oob"] = True
            
        # Timeout truncation
        elif self.step_count >= self.max_steps:
            truncated = True
            
        self.last_action = action.copy()
        obs = self._get_obs()
        
        return obs, float(reward), terminated, truncated, info
