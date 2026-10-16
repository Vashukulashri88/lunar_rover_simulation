"""
visualize_3d_mission.py
Generates full 3D perspective surface visualization of the lunar terrain,
craters, boulders, and autonomous rover trajectory under 1/6th lunar gravity.
Saves high-resolution 3D figures to results/lunar_mission_3d_surface.png.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from lunar_rover_physics import LunarTerrain, LunarRoverPhysics
from onboard_navigator import OnboardNeuralNavigator


def render_3d_lunar_mission(output_path="results/lunar_mission_3d_surface.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 1. Generate Expansive 10m x 10m Terrain
    terrain = LunarTerrain(arena_size=10.0, num_craters=12, num_boulders=18, seed=42)
    rover = LunarRoverPhysics(terrain)
    navigator = OnboardNeuralNavigator()
    
    # 2. Simulate Mission Episode across 10m Map
    start_x, start_y = -3.8, -3.8
    goal_x, goal_y = 3.6, 3.8
    terrain.ensure_safe_zones([(start_x, start_y), (goal_x, goal_y)], min_clearance=0.85)
    initial_yaw = np.arctan2(goal_y - start_y, goal_x - start_x)
    rover.reset((start_x, start_y, initial_yaw))
    
    traj_x, traj_y, traj_z = [rover.x], [rover.y], [rover.z]
    sensor_sample_rays = []
    
    for step in range(850):
        # Read rays and state
        rays = rover.get_raycast_observations()
        imu = rover.get_imu_state()
        
        dx = goal_x - rover.x
        dy = goal_y - rover.y
        dist = np.hypot(dx, dy)
        norm_dist = min(1.0, dist / (10.0 * np.sqrt(2.0)))
        target_heading = np.arctan2(dy, dx)
        heading_err = (target_heading - rover.yaw + np.pi) % (2 * np.pi) - np.pi
        
        obs = np.zeros(23, dtype=np.float32)
        obs[:15] = rays
        obs[15] = rover.pitch
        obs[16] = rover.roll
        obs[17] = rover.omega
        obs[18] = imu["linear_speed_ms"]
        obs[19] = rover.slip_ratio
        obs[20] = norm_dist
        obs[21] = np.sin(heading_err)
        obs[22] = np.cos(heading_err)
        
        action = navigator.predict_action(obs)
        rover.step(action[0], action[1], dt=0.05)
        
        traj_x.append(rover.x)
        traj_y.append(rover.y)
        traj_z.append(rover.z)
        
        if step % 50 == 0:
            sensor_sample_rays.append((rover.x, rover.y, rover.z, rover.yaw, list(rays)))
            
        if dist < 0.25 or rover.is_trapped_in_crater or rover.has_collided_boulder:
            break

    # 3. Build 3D Meshgrid over 10m x 10m Expanse
    res = 140
    x_grid = np.linspace(-5.0, 5.0, res)
    y_grid = np.linspace(-5.0, 5.0, res)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.zeros_like(X)
    
    for i in range(res):
        for j in range(res):
            Z[i, j] = terrain.get_height(X[i, j], Y[i, j])

    # 4. Plot 3D Scene
    fig = plt.figure(figsize=(14, 10), facecolor='#090d16')
    ax = fig.add_subplot(111, projection='3d', facecolor='#090d16')
    
    # 3D Surface with harsh lunar lighting
    surf = ax.plot_surface(
        X, Y, Z,
        cmap='copper',
        alpha=0.65,
        edgecolor='#334155',
        linewidth=0.15,
        antialiased=True,
        rstride=2,
        cstride=2
    )
    
    # Draw Craters in 3D (highlighting negative drop-off rims)
    for c in terrain.craters:
        theta = np.linspace(0, 2 * np.pi, 40)
        cx = c["x"] + c["r"] * np.cos(theta)
        cy = c["y"] + c["r"] * np.sin(theta)
        cz = np.array([terrain.get_height(x, y) for x, y in zip(cx, cy)])
        ax.plot(cx, cy, cz, color='#38bdf8', lw=2.2, ls='--', label='Crater Rim' if c == terrain.craters[0] else "")
        # Center marker
        cz_bottom = terrain.get_height(c["x"], c["y"])
        ax.scatter(c["x"], c["y"], cz_bottom, color='#0284c7', s=45)
        ax.text(c["x"], c["y"], cz_bottom - 0.04, f"Crater\n(-{c['d']*100:.0f}cm)", color='#bae6fd', fontsize=8, ha='center')

    # Draw Boulders in 3D (positive obstacles)
    for b in terrain.boulders:
        bz = terrain.get_height(b["x"], b["y"])
        # Hemisphere coordinates
        u = np.linspace(0, 2 * np.pi, 16)
        v = np.linspace(0, np.pi / 2, 8)
        bx = b["x"] + b["r"] * np.outer(np.cos(u), np.sin(v))
        by = b["y"] + b["r"] * np.outer(np.sin(u), np.sin(v))
        b_mesh_z = bz + b["h"] * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(bx, by, b_mesh_z, color='#d97706', alpha=0.9, edgecolor='#78350f', lw=0.5)

    # Draw 3D Trajectory
    ax.plot(traj_x, traj_y, [z + 0.015 for z in traj_z], color='#22c55e', lw=3.5, label='Autonomous Trajectory', zorder=10)
    
    # Start and Goal Markers
    start_z = terrain.get_height(start_x, start_y)
    goal_z = terrain.get_height(goal_x, goal_y)
    ax.scatter(start_x, start_y, start_z + 0.05, color='#3b82f6', s=160, marker='o', label='Start Landing Site', depthshade=False)
    ax.scatter(goal_x, goal_y, goal_z + 0.08, color='#eab308', s=240, marker='*', label='Exploration Goal Waypoint', depthshade=False)
    
    # Forward Sensor Ray Projections
    for rx, ry, rz, yaw, rays in sensor_sample_rays:
        for angle_deg in [-45, 0, 45]:
            rad = yaw + np.radians(angle_deg)
            ray_len = 0.6
            dx = ray_len * np.cos(rad)
            dy = ray_len * np.sin(rad)
            ax.plot([rx, rx + dx], [ry, ry + dy], [rz + 0.18, terrain.get_height(rx + dx, ry + dy)], color='#a855f7', lw=1.2, alpha=0.5)

    # Visual tuning
    ax.set_title("Autonomous Lunar Rover 3D Exploration & Hazard Avoidance\n(Procedural Lunar Regolith Dunes, Craters & Boulders under 1.62 m/s² Gravity)",
                 color='#f8fafc', fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel("X Arena (meters)", color='#94a3b8', labelpad=10)
    ax.set_ylabel("Y Arena (meters)", color='#94a3b8', labelpad=10)
    ax.set_zlabel("Elevation Z (meters)", color='#94a3b8', labelpad=10)
    
    ax.tick_params(colors='#94a3b8')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#1e293b')
    ax.yaxis.pane.set_edgecolor('#1e293b')
    ax.zaxis.pane.set_edgecolor('#1e293b')
    
    ax.view_init(elev=42, azim=-55)
    ax.legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='#f8fafc', loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[OK] 3D Lunar Mission Surface Visualization saved to: {output_path}")

if __name__ == "__main__":
    render_3d_lunar_mission()
