import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle

def get_user_choice():
    """Get user's choice for exit corner"""
    print("\nChoose the exit corner for the robot:")
    print("1. Top-Left (0, side_length)")
    print("2. Top-Right (side_length, side_length)")
    print("3. Bottom-Left (0, 0)")
    print("4. Bottom-Right (side_length, 0)")
    
    while True:
        try:
            choice = int(input("Enter your choice (1-4): "))
            if choice in [1, 2, 3, 4]:
                return choice
            else:
                print("Please enter a number between 1 and 4.")
        except ValueError:
            print("Please enter a valid number.")

def generate_path(n, exit_choice):
    """Generate path with complete farm coverage, ending at chosen corner"""
    total_columns = n + 2
    side_length = total_columns - 1
    
    path = []
    
    # Phase 1: Determine starting position based on exit choice and column parity
    total_columns_is_even = (total_columns % 2 == 0)
    
    if exit_choice == 1:  # Top-Left exit
        if total_columns_is_even:
            start_x = n
            start_y = side_length
        else:
            start_x = n
            start_y = 0
            
    elif exit_choice == 2:  # Top-Right exit
        if total_columns_is_even:
            start_x = 1
            start_y = side_length
        else:
            start_x = 1
            start_y = 0
            
    elif exit_choice == 3:  # Bottom-Left exit
        if total_columns_is_even:
            start_x = n
            start_y = 0
        else:
            start_x = n
            start_y = side_length
            
    elif exit_choice == 4:  # Bottom-Right exit
        if total_columns_is_even:
            start_x = 1
            start_y = 0
        else:
            start_x = 1
            start_y = side_length
    
    path.append((start_x, start_y))
    
    # Phase 2: Zigzag through inner columns
    if exit_choice in [1, 3]:  # Left exits
        column_range = list(range(start_x, 0, -1))  # Go from start_x down to 1
    else:  # Right exits
        column_range = list(range(start_x, n + 1))  # Go from start_x up to n
    
    current_y = start_y
    for i, col in enumerate(column_range):
        if current_y == 0:
            opposite_y = side_length
        else:
            opposite_y = 0
        path.append((col, opposite_y))
        current_y = opposite_y
        
        if i < len(column_range) - 1:
            if exit_choice in [1, 3]:  # Left exits
                next_col = col - 1
            else:  # Right exits
                next_col = col + 1
            path.append((next_col, current_y))
    
    # Phase 3: Cover the outer columns (0 and n+1) and navigate to exit
    last_x, last_y = path[-1]
    
    if exit_choice == 1:  # Top-Left (0, side_length)
        path.append((0, last_y))
        opposite_y = side_length if last_y == 0 else 0
        path.append((0, opposite_y))
        path.append((side_length, opposite_y))
        final_opposite_y = side_length if opposite_y == 0 else 0
        path.append((side_length, final_opposite_y))
        if final_opposite_y == 0:
            path.append((0, 0))
            path.append((0, side_length))
        else:
            path.append((0, side_length))
    
    elif exit_choice == 2:  # Top-Right (side_length, side_length)
        path.append((side_length, last_y))
        opposite_y = side_length if last_y == 0 else 0
        path.append((side_length, opposite_y))
        path.append((0, opposite_y))
        final_opposite_y = side_length if opposite_y == 0 else 0
        path.append((0, final_opposite_y))
        if final_opposite_y == 0:
            path.append((0, side_length))
            path.append((side_length, side_length))
        else:
            path.append((side_length, side_length))
    
    elif exit_choice == 3:  # Bottom-Left (0, 0)
        path.append((0, last_y))
        opposite_y = side_length if last_y == 0 else 0
        path.append((0, opposite_y))
        path.append((side_length, opposite_y))
        final_opposite_y = side_length if opposite_y == 0 else 0
        path.append((side_length, final_opposite_y))
        if final_opposite_y == side_length:
            path.append((0, side_length))
            path.append((0, 0))
        else:
            path.append((0, 0))
    
    elif exit_choice == 4:  # Bottom-Right (side_length, 0)
        path.append((side_length, last_y))
        opposite_y = side_length if last_y == 0 else 0
        path.append((side_length, opposite_y))
        path.append((0, opposite_y))
        final_opposite_y = side_length if opposite_y == 0 else 0
        path.append((0, final_opposite_y))
        if final_opposite_y == side_length:
            path.append((0, 0))
            path.append((side_length, 0))
        else:
            path.append((side_length, 0))
    
    return path

def interpolate_path(path, points_per_segment=20):
    """Create smooth interpolated path"""
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

def is_boundary_movement(path, seg_idx, n):
    """Check if a movement is actual boundary farming (final perimeter coverage only)"""
    if seg_idx >= len(path) - 1:
        return False
    
    x1, y1 = path[seg_idx]
    x2, y2 = path[seg_idx + 1]
    side_length = n + 1
    
    # Horizontal boundary farming
    if y1 == y2 and (y1 == 0 or y1 == side_length) and x1 != x2:
        if x1 == 0 or x1 == side_length or x2 == 0 or x2 == side_length:
            return True
    
    # Vertical boundary farming
    if x1 == x2 and (x1 == 0 or x1 == side_length) and y1 != y2:
        return True
    
    return False

def is_vertical_farming_movement(path, seg_idx, n):
    """Check if movement is vertical farming within a column (actual sowing)"""
    if seg_idx >= len(path) - 1:
        return False
    
    x1, y1 = path[seg_idx]
    x2, y2 = path[seg_idx + 1]
    return x1 == x2 and y1 != y2

def will_be_farmed_later(path, seg_idx, n):
    """Check if a path segment will be farmed later in the journey"""
    if seg_idx >= len(path) - 1:
        return False
    
    x1, y1 = path[seg_idx]
    x2, y2 = path[seg_idx + 1]
    side_length = n + 1
    
    # Horizontal boundary movement detection
    if y1 == y2 and (y1 == 0 or y1 == side_length) and x1 != x2:
        # Check future boundary passes on same line
        for future_idx in range(seg_idx + 1, len(path) - 1):
            fx1, fy1 = path[future_idx]
            fx2, fy2 = path[future_idx + 1]
            if fy1 == fy2 == y1 and is_boundary_movement(path, future_idx, n):
                return True
        return False
    
    # Inner horizontal transitions should stay unsown if future vertical farming exists in that column
    if y1 == y2 and y1 != 0 and y1 != side_length and x1 != x2:
        # Check if a vertical farming movement happens later in either column
        for future_idx in range(seg_idx + 1, len(path) - 1):
            fx1, fy1 = path[future_idx]
            fx2, fy2 = path[future_idx + 1]
            if (fx1 == fx2 == x1 and fy1 != fy2) or (fx1 == fx2 == x2 and fy1 != fy2):
                return True
        return False
    
    return False

def animate_robot(n, exit_choice):
    """Animate the robot following the path"""
    path = generate_path(n, exit_choice)
    x_coords, y_coords = zip(*path)
    smooth_x, smooth_y = interpolate_path(path, points_per_segment=25)
    points_per_segment = 25
    
    # Precompute which segments should be sown
    total_segments = len(path) - 1
    sow_segments = []
    for seg_idx in range(total_segments):
        should_sow = is_vertical_farming_movement(path, seg_idx, n) or is_boundary_movement(path, seg_idx, n)
        refarm = will_be_farmed_later(path, seg_idx, n)
        sow_segments.append(should_sow and not refarm)
    
    corner_names = {1: "Top-Left", 2: "Top-Right", 3: "Bottom-Left", 4: "Bottom-Right"}
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(-1, n + 3)
    ax.set_ylim(-1, n + 3)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.7)
    
    title = f'Farming Robot: {n+2} columns - Exit: {corner_names[exit_choice]}\nBrown=Unsown, Green=Sown(No-Go Zone)'
    ax.set_title(title, fontsize=14, pad=20)
    
    square = Rectangle((0, 0), n+1, n+1, fill=False, edgecolor='navy', linewidth=3, zorder=1)
    ax.add_patch(square)
    
    for i in range(0, n+2):
        ax.axvline(x=i, color='lightblue', alpha=0.5, linestyle='--', zorder=0)
    
    full_path, = ax.plot(x_coords, y_coords, color='#8B4513', alpha=0.8, linewidth=4, zorder=2)
    sown_path, = ax.plot([], [], color='#228B22', linewidth=6, alpha=0.9, zorder=3)
    
    robot = Circle((0, 0), 0.35, color='orange', ec='black', lw=2, zorder=4)
    ax.add_patch(robot)
    
    start_marker = Circle(path[0], 0.45, color='blue', fill=False, lw=3, zorder=5)
    end_marker = Circle(path[-1], 0.45, color='red', fill=False, lw=3, zorder=5)
    ax.add_patch(start_marker)
    ax.add_patch(end_marker)
    
    for i, (x, y) in enumerate(path):
        if i == 0:
            ax.text(x+0.5, y+0.2, f'START({x},{y})', fontsize=9, ha='left', va='bottom', 
                   color='blue', weight='bold', zorder=6)
        elif i == len(path)-1:
            ax.text(x+0.5, y+0.2, f'EXIT({x},{y})', fontsize=9, ha='left', va='bottom', 
                   color='red', weight='bold', zorder=6)
    
    ax.text(0.02, 0.98, 'Legend:\n🟤 Unsown Areas\n🟢 Sown Areas (No-Go)\n🟠 Farming Robot', 
            transform=ax.transAxes, fontsize=10, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    side_length = n + 1
    corners = [(0, 0), (side_length, 0), (0, side_length), (side_length, side_length)]
    corner_labels = ["BL", "BR", "TL", "TR"]
    
    for (cx, cy), label in zip(corners, corner_labels):
        marker_color = 'red' if (cx, cy) == path[-1] else 'gray'
        corner_circle = Circle((cx, cy), 0.2, color=marker_color, alpha=0.7, zorder=5)
        ax.add_patch(corner_circle)
        ax.text(cx-0.3, cy-0.3, label, fontsize=8, ha='center', va='center', 
                color='red' if (cx, cy) == path[-1] else 'gray', weight='bold')

    def init():
        sown_path.set_data([], [])
        robot.center = (smooth_x[0], smooth_y[0])
        return sown_path, robot

    def update(frame):
        if frame >= len(smooth_x):
            frame = len(smooth_x) - 1
        
        robot.center = (smooth_x[frame], smooth_y[frame])
        
        sown_x = []
        sown_y = []
        
        points_per_segment = 25
        
        for seg_idx in range(len(path) - 1):
            seg_start_smooth = seg_idx * points_per_segment
            seg_end_smooth = min((seg_idx + 1) * points_per_segment, len(smooth_x))
            
            # Check if this segment should be sown using precomputed information
            is_sowing = sow_segments[seg_idx]
            
            if frame > seg_end_smooth and is_sowing:
                segment_x = smooth_x[seg_start_smooth:seg_end_smooth]
                segment_y = smooth_y[seg_start_smooth:seg_end_smooth]
                sown_x.extend(segment_x)
                sown_x.append(np.nan)
                sown_y.extend(segment_y)
                sown_y.append(np.nan)
            elif frame > seg_start_smooth and is_sowing:
                partial_end = min(frame, seg_end_smooth)
                segment_x_partial = smooth_x[seg_start_smooth:partial_end]
                segment_y_partial = smooth_y[seg_start_smooth:partial_end]
                sown_x.extend(segment_x_partial)
                sown_y.extend(segment_y_partial)
        
        sown_path.set_data(sown_x, sown_y)
        
        return sown_path, robot
    
    ani = FuncAnimation(fig, update, frames=len(smooth_x), init_func=init, blit=True, interval=50)
    
    print(f"\nPath Summary:")
    print(f"Total waypoints: {len(path)}")
    print(f"Start: {path[0]}")
    print(f"End: {path[-1]}")
    print(f"Columns covered: 0 to {n+1} (all {n+2} columns)")
    
    columns_visited = set(x for x, y in path)
    print(f"Columns visited: {sorted(columns_visited)}")
    if len(columns_visited) == n + 2:
        print("✓ All columns covered!")
    else:
        missing = set(range(n+2)) - columns_visited
        print(f"✗ Missing columns: {missing}")
    
    print(f"\nDetailed path with movement types:")
    for i, (x, y) in enumerate(path):
        if i < len(path) - 1:
            next_x, next_y = path[i + 1]
            will_refarm = will_be_farmed_later(path, i, n)
            if is_vertical_farming_movement(path, i, n):
                move_type = "VERTICAL FARMING (sown)"
            elif is_boundary_movement(path, i, n):
                move_type = "BOUNDARY FARMING (sown)"
            else:
                move_type = "TRANSITION (unsown)"
            if will_refarm:
                move_type += " [WILL BE RE-FARMED - STAYS BROWN]"
            print(f"Step {i+1}: ({x}, {y}) -> ({next_x}, {next_y}) [{move_type}]")
        else:
            print(f"Step {i+1}: ({x}, {y}) [END]")
    
    plt.tight_layout()
    plt.show()
    return ani

def main():
    """Main function to run the interactive robot path generator"""
    print("=== Fixed Robot Path Generator ===")
    print("The robot will now cover ALL columns and navigate efficiently to any exit.")
    
    while True:
        try:
            n = int(input("Enter the number of inner columns (n): "))
            if n > 0:
                break
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")
    
    exit_choice = get_user_choice()
    
    print(f"\nGenerating path for {n+2} total columns with exit at corner {exit_choice}...")
    print("The robot will cover ALL columns (0 to n+1) efficiently.")
    
    ani = animate_robot(n, exit_choice)
    
    return ani

if __name__ == "__main__":
    ani = main()
