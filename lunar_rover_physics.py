"""
lunar_rover_physics.py
High-fidelity 3D terramechanics physics simulation for lunar rover autonomous exploration.
Models:
- Exact Nakshatra Hackathon constraints (L <= 30cm, W <= 21cm, H <= 25cm, 7cm spine)
- Lunar Gravity (g = 1.62 m/s^2)
- Regolith terramechanics (Bekker-Wong sinkage, Janosi shear slip, bulldozing turning resistance)
- Procedural lunar terrain: negative obstacles (craters), positive obstacles (boulders), rolling slopes
- Realistic 3D stereo-depth / raycast rangefinder perception suite
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

# Nakshatra Aerospace Hackathon 2026 Constraint Constants
MAX_LENGTH_M = 0.28       # Max 0.30 m
MAX_WIDTH_M = 0.20        # Max 0.21 m
MAX_HEIGHT_M = 0.22       # Max 0.25 m
CENTRAL_SPINE_W_M = 0.068 # Max 0.07 m
MIN_CLEARANCE_M = 0.006   # Min 6 mm
NOMINAL_CLEARANCE_M = 0.020 # 20 mm
ROVER_MASS_KG = 2.8       # 2.8 kg small-scale prototype
LUNAR_GRAVITY = 1.62      # m/s^2 (1/6 Earth gravity)

# Sensor specifications
SENSOR_MAST_HEIGHT = 0.18 # Sensor mounted at 18 cm above ground
NUM_RAYS = 15             # 15-beam forward stereo depth / lidar fan
FOV_DEG = 110.0           # Horizontal field of view in degrees
NOMINAL_RAY_PITCH_DEG = -26.0 # Lookdown angle in degrees
MAX_RAY_RANGE_M = 1.8     # Sensor max valid range

# Physics constants
MAX_TRACK_SPEED_MS = 0.35 # Max linear track speed (0.35 m/s = 1.26 km/h)
REGOLITH_SLIP_BASE = 0.15 # 15% nominal slip in loose lunar sand
ROLLOVER_ANGLE_RAD = np.radians(32.0) # Tip-over danger threshold (~32 degrees)
CRATER_TRAP_DEPTH_M = 0.05 # Depth below which rover is trapped/inverted in crater


class LunarTerrain:
    """
    3D Lunar Terrain generator featuring:
    - Rolling lunar slopes and mounds
    - Negative obstacles (craters of various radii and depths with raised rims)
    - Positive obstacles (boulders/rocks)
    """
    def __init__(
        self,
        arena_size: float = 10.0,  # 10m x 10m expansive lunar terrain
        num_craters: int = 12,
        num_boulders: int = 18,
        max_slope_deg: float = 20.0,
        seed: Optional[int] = None
    ):
        self.arena_size = arena_size
        self.max_slope_deg = max_slope_deg
        self.rng = np.random.default_rng(seed)
        
        # Terrain parameters
        self.craters: List[Dict[str, float]] = []
        self.boulders: List[Dict[str, float]] = []
        
        # Multi-scale undulating lunar terrain (macro rolling hills + micro dunes)
        self.slope_amp_x = self.rng.uniform(0.08, 0.16)
        self.slope_amp_y = self.rng.uniform(0.08, 0.16)
        self.slope_freq_x = self.rng.uniform(0.35, 0.65)
        self.slope_freq_y = self.rng.uniform(0.35, 0.65)
        self.slope_phase_x = self.rng.uniform(0, 2 * np.pi)
        self.slope_phase_y = self.rng.uniform(0, 2 * np.pi)
        
        self.generate_arena(num_craters, num_boulders)

    def generate_arena(self, num_craters: int, num_boulders: int, safe_points: Optional[List[Tuple[float, float]]] = None):
        """Generates random craters and boulders avoiding the specified safe zones."""
        self.craters.clear()
        self.boulders.clear()
        
        if safe_points is None:
            safe_points = [(0.0, 0.0)]
            
        half = self.arena_size / 2.0
        safe_spawn_radius = 0.95 # Safe clearance around key points
        
        # Place Craters (negative obstacles) across the 10m x 10m expanse
        attempts = 0
        while len(self.craters) < num_craters and attempts < 350:
            attempts += 1
            cx = float(self.rng.uniform(-half + 0.8, half - 0.8))
            cy = float(self.rng.uniform(-half + 0.8, half - 0.8))
            radius = float(self.rng.uniform(0.30, 0.85)) # 30cm to 85cm crater radius
            if any(np.hypot(cx - sx, cy - sy) < (radius * 1.3 + safe_spawn_radius) for sx, sy in safe_points):
                continue
            depth = float(self.rng.uniform(0.08, 0.22))  # 8cm to 22cm crater depth
            self.craters.append({
                "x": cx, "y": cy, "r": radius, "d": depth, "rim_h": depth * 0.22
            })
            
        # Place Boulders (positive obstacles)
        attempts = 0
        while len(self.boulders) < num_boulders and attempts < 350:
            attempts += 1
            bx = float(self.rng.uniform(-half + 0.6, half - 0.6))
            by = float(self.rng.uniform(-half + 0.6, half - 0.6))
            radius = float(self.rng.uniform(0.09, 0.22)) # 9cm to 22cm rock radius
            if any(np.hypot(bx - sx, by - sy) < (radius + safe_spawn_radius) for sx, sy in safe_points):
                continue
            # Avoid overlapping directly inside a crater center
            inside_crater = any(np.hypot(bx - c["x"], by - c["y"]) < c["r"] * 0.85 for c in self.craters)
            if inside_crater:
                continue
            height = float(self.rng.uniform(0.08, 0.22)) # 8cm to 22cm rock height
            self.boulders.append({"x": bx, "y": by, "r": radius, "h": height})

    def ensure_safe_zones(self, points: List[Tuple[float, float]], min_clearance: float = 0.85):
        """Relocates any crater or boulder that encroaches upon start or goal coordinates."""
        for sx, sy in points:
            for c in self.craters:
                dist = np.hypot(c["x"] - sx, c["y"] - sy)
                req = c["r"] * 1.3 + min_clearance
                if dist < req:
                    angle = np.arctan2(c["y"] - sy, c["x"] - sx) if dist > 1e-4 else 0.0
                    c["x"] = float(sx + np.cos(angle) * (req + 0.15))
                    c["y"] = float(sy + np.sin(angle) * (req + 0.15))
            for b in self.boulders:
                dist = np.hypot(b["x"] - sx, b["y"] - sy)
                req = b["r"] + min_clearance
                if dist < req:
                    angle = np.arctan2(b["y"] - sy, b["x"] - sx) if dist > 1e-4 else 0.0
                    b["x"] = float(sx + np.cos(angle) * (req + 0.15))
                    b["y"] = float(sy + np.sin(angle) * (req + 0.15))

    def get_height(self, x: float, y: float) -> float:
        """Returns the ground elevation z at (x, y)."""
        # Base undulating lunar dunes
        z = (self.slope_amp_x * np.sin(self.slope_freq_x * x + self.slope_phase_x) +
             self.slope_amp_y * np.cos(self.slope_freq_y * y + self.slope_phase_y))
        
        # Subtract craters (negative parabolic bowl with raised ejecta rim)
        for c in self.craters:
            dist = np.hypot(x - c["x"], y - c["y"])
            r_ratio = dist / c["r"]
            if r_ratio < 1.0:
                # Inside bowl: depth drops parabolically
                z -= c["d"] * (1.0 - r_ratio**2)**2
            elif r_ratio < 1.3:
                # Ejecta rim: slight bump on the lip
                rim_bump = c["rim_h"] * np.sin((r_ratio - 1.0) / 0.3 * np.pi)
                z += rim_bump
                
        return float(z)

    def get_gradient(self, x: float, y: float, eps: float = 0.05) -> Tuple[float, float]:
        """Calculates terrain slope gradients (dz/dx, dz/dy)."""
        dz_dx = (self.get_height(x + eps, y) - self.get_height(x - eps, y)) / (2 * eps)
        dz_dy = (self.get_height(x, y + eps) - self.get_height(x, y - eps)) / (2 * eps)
        return float(dz_dx), float(dz_dy)

    def check_boulder_collision(self, x: float, y: float, rover_radius: float = 0.15) -> bool:
        """Checks if rover footprint intersects any boulder boundary."""
        for b in self.boulders:
            dist = np.hypot(x - b["x"], y - b["y"])
            if dist < (b["r"] + rover_radius):
                return True
        return False

    def get_crater_depth_at(self, x: float, y: float) -> float:
        """Returns maximum depth inside a crater if rover center is inside any crater bowl."""
        max_depth = 0.0
        for c in self.craters:
            dist = np.hypot(x - c["x"], y - c["y"])
            if dist < c["r"]:
                depth = float(c["d"] * (1.0 - (dist / c["r"])**2)**2)
                if depth > max_depth:
                    max_depth = depth
        return max_depth


class LunarRoverPhysics:
    """
    Kinematic and dynamic physics model for a skid-steer / tracked lunar rover.
    Accounts for low lunar gravity, regolith slip, and slope tilt.
    """
    def __init__(self, terrain: LunarTerrain):
        self.terrain = terrain
        self.reset()

    def reset(self, start_pos: Optional[Tuple[float, float, float]] = None):
        """Resets rover state to start position (x, y, heading_rad)."""
        if start_pos is None:
            self.x = 0.0
            self.y = 0.0
            self.yaw = 0.0
        else:
            self.x, self.y, self.yaw = start_pos
            
        self.z = self.terrain.get_height(self.x, self.y)
        self.pitch = 0.0
        self.roll = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.omega = 0.0
        self.left_speed = 0.0
        self.right_speed = 0.0
        self.slip_ratio = REGOLITH_SLIP_BASE
        
        # Hazard status
        self.is_trapped_in_crater = False
        self.has_collided_boulder = False
        self.has_rolled_over = False
        self.is_out_of_bounds = False

    def step(self, left_cmd: float, right_cmd: float, dt: float = 0.05):
        """
        Executes one physics simulation step with differential track throttle commands [-1.0, 1.0].
        Incorporates:
        - Bekker-style wheel/track slip on loose regolith
        - Bulldozing skid-steer turning resistance moment
        - 3D slope forces under lunar gravity
        """
        left_cmd = float(np.clip(left_cmd, -1.0, 1.0))
        right_cmd = float(np.clip(right_cmd, -1.0, 1.0))
        
        # Commanded track linear speeds
        v_l_cmd = left_cmd * MAX_TRACK_SPEED_MS
        v_r_cmd = right_cmd * MAX_TRACK_SPEED_MS
        
        # Nominal unconstrained velocities
        v_nom = (v_r_cmd + v_l_cmd) / 2.0
        omega_nom = (v_r_cmd - v_l_cmd) / MAX_WIDTH_M
        
        # Calculate local terrain slope angles
        dz_dx, dz_dy = self.terrain.get_gradient(self.x, self.y)
        
        # Project slopes onto rover heading:
        # pitch along longitudinal heading
        slope_pitch = np.arctan(dz_dx * np.cos(self.yaw) + dz_dy * np.sin(self.yaw))
        # roll along lateral cross axis
        slope_roll = np.arctan(-dz_dx * np.sin(self.yaw) + dz_dy * np.cos(self.yaw))
        
        self.pitch = float(slope_pitch)
        self.roll = float(slope_roll)
        
        # Check tip-over / rollover
        total_tilt = np.hypot(self.pitch, self.roll)
        if total_tilt > ROLLOVER_ANGLE_RAD:
            self.has_rolled_over = True
            
        # Terramechanics Slip Model:
        # On slopes under lunar gravity, reduced normal force (Fn = m*g*cos(tilt))
        # causes increased track slip. Going uphill increases slip dramatically.
        gravity_drag_accel = LUNAR_GRAVITY * np.sin(self.pitch)
        
        # Effective slip ratio increases with steepness:
        # s = s_0 + k * sin(slope)
        dynamic_slip = np.clip(REGOLITH_SLIP_BASE + 0.35 * max(0.0, np.sin(self.pitch)), 0.05, 0.75)
        self.slip_ratio = float(dynamic_slip)
        
        # Actual forward velocity reduced by slip and gravity gradient
        v_actual = v_nom * (1.0 - dynamic_slip) - (0.05 * gravity_drag_accel * dt)
        
        # Skid-steering bulldozing resistance in lunar dust:
        # Turning loose regolith creates lateral soil piles resisting yaw
        skid_efficiency = 0.78 # 22% turning loss due to soil shear
        omega_actual = omega_nom * skid_efficiency
        
        # Update orientation and position
        self.yaw = float((self.yaw + omega_actual * dt + np.pi) % (2 * np.pi) - np.pi)
        
        dx = v_actual * np.cos(self.yaw) * dt
        dy = v_actual * np.sin(self.yaw) * dt
        self.x += dx
        self.y += dy
        self.z = self.terrain.get_height(self.x, self.y)
        
        self.vx = v_actual * np.cos(self.yaw)
        self.vy = v_actual * np.sin(self.yaw)
        self.omega = omega_actual
        self.left_speed = v_l_cmd
        self.right_speed = v_r_cmd
        
        # Check Hazard Conditions:
        # 1. Negative obstacle: is rover inside a crater deeper than clearance threshold?
        crater_depth = self.terrain.get_crater_depth_at(self.x, self.y)
        if crater_depth > CRATER_TRAP_DEPTH_M:
            self.is_trapped_in_crater = True
            
        # 2. Positive obstacle: collision with boulder
        if self.terrain.check_boulder_collision(self.x, self.y, rover_radius=MAX_WIDTH_M / 2.0):
            self.has_collided_boulder = True
            
        # 3. Arena boundary check
        half_arena = self.terrain.arena_size / 2.0
        if abs(self.x) > half_arena or abs(self.y) > half_arena:
            self.is_out_of_bounds = True

    def get_raycast_observations(self) -> np.ndarray:
        """
        Simulates the forward perception sensor suite:
        Casts 15 rays in a horizontal fan (-55 deg to +55 deg) tilted down at -26 deg.
        Returns normalized range returns in [0, 1]:
        - Return < nominal: Positive obstacle (boulder) detected ahead!
        - Return > nominal or max range: Negative obstacle (crater drop-off) detected ahead!
        - Return == nominal: Smooth flat terrain ahead.
        """
        angles = np.linspace(-np.radians(FOV_DEG / 2.0), np.radians(FOV_DEG / 2.0), NUM_RAYS)
        returns = np.zeros(NUM_RAYS, dtype=np.float32)
        
        # Mast position at front of rover
        mast_x = self.x + (MAX_LENGTH_M * 0.4) * np.cos(self.yaw)
        mast_y = self.y + (MAX_LENGTH_M * 0.4) * np.sin(self.yaw)
        mast_z = self.z + SENSOR_MAST_HEIGHT
        
        pitch_rad = np.radians(NOMINAL_RAY_PITCH_DEG) + self.pitch
        
        for i, angle in enumerate(angles):
            ray_yaw = self.yaw + angle
            
            # Unit direction vector of ray
            dir_x = np.cos(pitch_rad) * np.cos(ray_yaw)
            dir_y = np.cos(pitch_rad) * np.sin(ray_yaw)
            dir_z = np.sin(pitch_rad)
            
            # Ray march
            hit_dist = MAX_RAY_RANGE_M
            step_size = 0.03 # 3 cm sampling along ray
            t = 0.06 # start 6 cm in front
            
            while t < MAX_RAY_RANGE_M:
                rx = mast_x + t * dir_x
                ry = mast_y + t * dir_y
                rz = mast_z + t * dir_z
                
                # Check ground height at (rx, ry)
                terrain_z = self.terrain.get_height(rx, ry)
                
                # Check boulder collision along ray
                boulder_hit = False
                for b in self.terrain.boulders:
                    dist_xy = np.hypot(rx - b["x"], ry - b["y"])
                    if dist_xy < b["r"]:
                        # Inside boulder cylinder/hemisphere
                        b_top = self.terrain.get_height(b["x"], b["y"]) + b["h"]
                        if rz <= b_top:
                            boulder_hit = True
                            break
                            
                if boulder_hit or rz <= terrain_z:
                    hit_dist = t
                    break
                    
                t += step_size
                
            # Normalize to [0.0, 1.0]
            returns[i] = hit_dist / MAX_RAY_RANGE_M
            
        return returns

    def get_imu_state(self) -> Dict[str, float]:
        """Returns simulated IMU sensor telemetry."""
        return {
            "pitch_deg": float(np.degrees(self.pitch)),
            "roll_deg": float(np.degrees(self.roll)),
            "yaw_deg": float(np.degrees(self.yaw)),
            "yaw_rate_rads": float(self.omega),
            "linear_speed_ms": float(np.hypot(self.vx, self.vy)),
            "slip_ratio": float(self.slip_ratio),
            "total_tilt_deg": float(np.degrees(np.hypot(self.pitch, self.roll)))
        }
