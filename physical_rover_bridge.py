"""
physical_rover_bridge.py
Hardware Integration Bridge for Team Batman's Lunar Rover Prototype.
Nakshatra Aerospace Hackathon 2026 - Problem Statement 01.

Connects:
1. Stereo Depth / Dual Cameras (or Ultrasonic / ToF distance array) -> 15-Beam observation fan
2. MPU-6050 / BNO055 IMU -> Pitch, Roll, Yaw rate
3. OnboardNeuralNavigator (Trained PPO Edge AI Model) -> Predicts optimal track velocities
4. Hardware Actuation -> Serial / PWM commands to Tank Tread Motor Drivers & 4-Leg Active Servos
"""

import time
import math
import numpy as np

# Import the lightweight zero-dependency edge neural policy and LLM co-pilot
from onboard_navigator import OnboardNeuralNavigator
from lunar_rover_llm_copilot import LunarRoverLLMCopilot

class PhysicalLunarRoverBridge:
    def __init__(self, serial_port: str = "/dev/ttyUSB0", baud_rate: int = 115200, mock_hardware: bool = True):
        self.mock_hardware = mock_hardware
        self.navigator = OnboardNeuralNavigator()
        self.copilot = LunarRoverLLMCopilot()
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        
        # State estimation
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.target_x = 2.0
        self.target_y = 2.0
        
        # Team Batman Active Suspension Trimming
        self.suspension_clearance_mm = 20.0 # Nominal 20mm (Range: 6mm - 35mm)
        self.last_llm_step_time = 0.0
        self.latest_llm_decision = {}
        
        print(f"=== Physical Rover Hardware Bridge Initialized ===")
        print(f"Hardware Mock Mode: {self.mock_hardware}")
        print(f"Edge PPO Policy: Loaded ({len(self.navigator.b1)} hidden units)")
        print(f"Cognitive LLM Co-Pilot: Active ({self.copilot.backend.upper()})")

    def read_imu(self) -> dict:
        """Reads MPU-6050 / BNO055 6-DOF IMU over I2C."""
        if self.mock_hardware:
            # Simulated nominal IMU reading
            return {
                "pitch_rad": float(np.radians(-1.5)),
                "roll_rad": float(np.radians(2.0)),
                "yaw_rate_rads": 0.02,
                "linear_speed_ms": 0.22,
                "slip_ratio": 0.16
            }
        else:
            # Physical I2C reading logic goes here (e.g. smbus2 or adafruit_bno055)
            pass

    def read_stereo_depth_rays(self) -> np.ndarray:
        """
        Processes stereo disparity image or 15-beam forward range array.
        Returns 15 normalized distance returns in [0, 1] (nominal flat ground ~0.228).
        """
        if self.mock_hardware:
            # Simulate nominal terrain (~0.228) with a boulder on the left (indices 11:13) and crater ahead (7:9)
            rays = np.full(15, 0.228, dtype=np.float32)
            rays[11:13] = 0.12 # Positive boulder detected at 21cm on left (+angle)
            rays[7:9] = 0.42   # Deep drop-off / crater detected straight ahead (>0.28)
            return rays
        else:
            # OpenCV StereoBM / StereoSGBM or DepthAI pipeline
            pass

    def compute_active_suspension(self, pitch_rad: float, roll_rad: float, target_clearance: Optional[float] = None) -> dict:
        """
        Team Batman Active 4-Leg Servo Suspension:
        f(alpha) -> beta, gamma and f(z) -> A, C
        Dynamically tilts chassis to counteract pitch/roll slopes and adjusts clearance (6mm - 35mm).
        """
        tilt_deg = math.degrees(math.hypot(pitch_rad, roll_rad))
        if target_clearance is None:
            target_clearance = self.suspension_clearance_mm
        # Elevate clearance automatically if steep slope encountered and LLM hasn't commanded low CG
        if tilt_deg > 12.0 and target_clearance == 20.0:
            target_clearance = 28.0
            
        # Servo differential offsets
        fl_servo = target_clearance - math.degrees(pitch_rad) * 0.4 + math.degrees(roll_rad) * 0.4
        fr_servo = target_clearance - math.degrees(pitch_rad) * 0.4 - math.degrees(roll_rad) * 0.4
        rl_servo = target_clearance + math.degrees(pitch_rad) * 0.4 + math.degrees(roll_rad) * 0.4
        rr_servo = target_clearance + math.degrees(pitch_rad) * 0.4 - math.degrees(roll_rad) * 0.4
        
        return {
            "clearance_mm": target_clearance,
            "servos": [float(np.clip(fl_servo, 6, 35)), float(np.clip(fr_servo, 6, 35)),
                       float(np.clip(rl_servo, 6, 35)), float(np.clip(rr_servo, 6, 35))]
        }

    def run_control_cycle(self):
        """Executes one real-time edge control loop (20 Hz / 50ms) with 1 Hz LLM tactical supervision."""
        # 1. Read Perception & IMU
        rays = self.read_stereo_depth_rays()
        imu = self.read_imu()
        
        # 2. High-Level LLM Tactical Evaluation (1 Hz cognitive cycle)
        now = time.time()
        if (now - self.last_llm_step_time) >= 1.0 or not self.latest_llm_decision:
            self.last_llm_step_time = now
            llm_eval = self.copilot.evaluate_tactical_step(
                rover_pos=(self.current_x, self.current_y, self.current_yaw),
                goal_pos=(self.target_x, self.target_y),
                imu_state={
                    "pitch_deg": math.degrees(imu["pitch_rad"]),
                    "roll_deg": math.degrees(imu["roll_rad"]),
                    "total_tilt_deg": math.degrees(math.hypot(imu["pitch_rad"], imu["roll_rad"])),
                    "slip_ratio": imu["slip_ratio"],
                    "linear_speed_ms": imu["linear_speed_ms"]
                },
                rays=rays.tolist()
            )
            self.latest_llm_decision = llm_eval.get("decision", {})
            self.suspension_clearance_mm = llm_eval.get("target_clearance_mm", 20.0)
        
        # 3. Compute Relative Target Vector
        dx = self.target_x - self.current_x
        dy = self.target_y - self.current_y
        dist = math.hypot(dx, dy)
        norm_dist = min(1.0, dist / 5.0)
        target_angle = math.atan2(dy, dx)
        heading_err = (target_angle - self.current_yaw + math.pi) % (2 * math.pi) - math.pi
        
        # 4. Assemble 23-dim Observation Vector
        obs = np.zeros(23, dtype=np.float32)
        obs[:15] = rays
        obs[15] = imu["pitch_rad"]
        obs[16] = imu["roll_rad"]
        obs[17] = imu["yaw_rate_rads"]
        obs[18] = imu["linear_speed_ms"]
        obs[19] = imu["slip_ratio"]
        obs[20] = norm_dist
        obs[21] = math.sin(heading_err)
        obs[22] = math.cos(heading_err)
        
        # 5. Neural Network Low-Level Inference (< 0.2ms PPO Actor)
        action = self.navigator.predict_action(obs)
        left_throttle, right_throttle = float(action[0]), float(action[1])
        
        # Apply LLM Speed Cap if active
        left_throttle *= self.copilot.speed_limit
        right_throttle *= self.copilot.speed_limit
        
        # 6. Dynamic Active Suspension Adjustment (governed by LLM target clearance)
        suspension = self.compute_active_suspension(imu["pitch_rad"], imu["roll_rad"])
        
        # 7. Dispatch Motor PWM Commands
        left_pwm = int(abs(left_throttle) * 255)
        right_pwm = int(abs(right_throttle) * 255)
        left_dir = 1 if left_throttle >= 0 else 0
        right_dir = 1 if right_throttle >= 0 else 0
        
        return {
            "left_throttle": left_throttle,
            "right_throttle": right_throttle,
            "left_pwm": left_pwm,
            "right_pwm": right_pwm,
            "suspension_clearance_mm": self.suspension_clearance_mm,
            "suspension_servos": suspension["servos"],
            "dist_to_goal_m": dist,
            "llm_hazard_level": self.latest_llm_decision.get("hazard_level", "NOMINAL"),
            "llm_reasoning": self.latest_llm_decision.get("reasoning", ""),
            "llm_action": self.latest_llm_decision.get("recommended_action", ""),
            "llm_downlink": self.latest_llm_decision.get("downlink_message", "")
        }

if __name__ == "__main__":
    bridge = PhysicalLunarRoverBridge(mock_hardware=True)
    print("\nExecuting Physical Rover Control Loop Iteration:")
    cmd = bridge.run_control_cycle()
    print(f"Left Track Motor:   PWM {cmd['left_pwm']:3d} (Throttle: {cmd['left_throttle']:+.2f})")
    print(f"Right Track Motor:  PWM {cmd['right_pwm']:3d} (Throttle: {cmd['right_throttle']:+.2f})")
    print(f"Active Suspension:  {cmd['suspension_clearance_mm']:.1f} mm Clearance (Servos: {[round(s,1) for s in cmd['suspension_servos']]})")
    print(f"Distance to Goal:   {cmd['dist_to_goal_m']:.2f} m")
