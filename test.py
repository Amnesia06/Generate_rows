import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle

def generate_path(n):
    total_columns = n + 2
    side_length = total_columns - 1
    is_odd = (total_columns % 2 != 0)
    
    path = []
    
    if is_odd:
        path.append((1, 0))
    else:
        path.append((1, side_length))
    
    if is_odd:
        for col in range(1, total_columns):
            if col % 2 == 1:
                path.append((col, side_length))
                if col < total_columns - 1:
                    path.append((col + 1, side_length))
            else:
                path.append((col, 0))
                if col < total_columns - 1:
                    path.append((col + 1, 0))
    else:
        for col in range(1, total_columns):
            if col % 2 == 1:
                path.append((col, 0))
                if col < total_columns - 1:
                    path.append((col + 1, 0))
            else:
                path.append((col, side_length))
                if col < total_columns - 1:
                    path.append((col + 1, side_length))
    
    last_x, last_y = path[-1]
    if last_x != 0:
        path.append((0, last_y))
    if last_y != side_length:
        path.append((0, side_length))
    if path[-1][0] != side_length:
        path.append((side_length, side_length))

    return path

def interpolate_path(path, points_per_segment=20):
    smooth_x = []
    smooth_y = []
    for i in range(len(path)-1):
        x_start, y_start = path[i]
        x_end, y_end = path[i+1]
        x_vals = np.linspace(x_start, x_end, points_per_segment)
        y_vals = np.linspace(y_start, y_end, points_per_segment)
        smooth_x.extend(x_vals)
        smooth_y.extend(y_vals)
    return np.array(smooth_x), np.array(smooth_y)

def animate_robot(n):
    path = generate_path(n)
    x_coords, y_coords = zip(*path)
    smooth_x, smooth_y = interpolate_path(path, points_per_segment=25)  # smoothness control
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(-1, n + 3)
    ax.set_ylim(-1, n + 3)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.7)
    title = f'Robot Path: {n+2} columns ({"Odd" if (n+2)%2 else "Even"})'
    ax.set_title(title, fontsize=14, pad=20)
    
    square = Rectangle((0, 0), n+1, n+1, fill=False, edgecolor='navy', linewidth=3, zorder=1)
    ax.add_patch(square)
    
    full_path, = ax.plot([], [], 'r-', alpha=0.2, linewidth=2, zorder=2)
    footprint, = ax.plot([], [], color='#006400', linewidth=4, alpha=0.8, zorder=3)
    
    robot = Circle((0, 0), 0.35, color='lime', ec='black', lw=2, zorder=4)
    ax.add_patch(robot)
    
    start_marker = Circle(path[0], 0.45, color='lime', fill=False, lw=3, zorder=5)
    end_marker = Circle(path[-1], 0.45, color='red', fill=False, lw=3, zorder=5)
    ax.add_patch(start_marker)
    ax.add_patch(end_marker)
    
    for i, (x, y) in enumerate(path):
        if i % 2 == 0 or i == len(path)-1:
            ax.text(x+0.1, y+0.1, f'({x},{y})', fontsize=10, ha='left', va='bottom', zorder=6)

    def init():
        full_path.set_data([], [])
        footprint.set_data([], [])
        robot.center = (smooth_x[0], smooth_y[0])
        return full_path, footprint, robot

    def update(frame):
        full_path.set_data(smooth_x[:frame+1], smooth_y[:frame+1])
        
        tail_length = min(40, frame+1)  # make tail longer for smooth
        footprint.set_data(smooth_x[frame+1-tail_length:frame+1], smooth_y[frame+1-tail_length:frame+1])
        
        if frame < len(smooth_x):
            robot.center = (smooth_x[frame], smooth_y[frame])
        
        return full_path, footprint, robot
    
    ani = FuncAnimation(fig, update, frames=len(smooth_x), init_func=init, blit=True, interval=20)
    
    plt.tight_layout()
    plt.show()
    return ani

# Run
animate_robot(n=6)  # Example
