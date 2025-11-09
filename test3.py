import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D
import csv
from datetime import datetime
import os
import math
import traceback
# --- Helper Functions ---
SOWN_SEGMENTS_LOG = set()


def _commit_smart_transition_sweep(points_list, sow_flags_list, curr_x, start_y, end_y, exit_lanes, gap_size=1):
    """
    Smart transition sweep that optimizes sowing vs positioning based on exit proximity.
    """
    if start_y == end_y:
        return end_y
    
    ex_ln_x, ex_ln_y = exit_lanes
    direction = 1 if end_y > start_y else -1
    total_distance = abs(end_y - start_y)
    
    # Calculate distance to exit from current position vs end position
    dist_from_start = abs(curr_x - ex_ln_x) + abs(start_y - ex_ln_y)
    dist_from_end = abs(curr_x - ex_ln_x) + abs(end_y - ex_ln_y)
    
    # If we're getting closer to exit, sow more; if moving away, sow less
    if dist_from_end < dist_from_start:
        # Moving closer to exit - sow normally with small gaps
        sow_ratio = 0.9
    else:
        # Moving away from exit - sow less, save time for positioning
        sow_ratio = 0.6
    
    # Calculate sowing distance based on optimization ratio
    sow_distance = int(total_distance * sow_ratio)
    
    if sow_distance <= gap_size * 2:
        # Too short for meaningful sowing, just position
        _commit_point_to_path(points_list, sow_flags_list, (curr_x, end_y), False, "SmartTransition_Position")
        return end_y
    
    # Phase 1: Initial gap
    gap_end_y = start_y + (gap_size * direction)
    if gap_end_y != start_y:
        _commit_point_to_path(points_list, sow_flags_list, (curr_x, gap_end_y), False, "SmartTransition_InitialGap")
    
    # Phase 2: Optimized sowing
    sow_end_y = start_y + (sow_distance * direction)
    _commit_point_to_path(points_list, sow_flags_list, (curr_x, sow_end_y), True, "SmartTransition_OptimizedSow")
    
    # Phase 3: Positioning to end
    if sow_end_y != end_y:
        _commit_point_to_path(points_list, sow_flags_list, (curr_x, end_y), False, "SmartTransition_FinalPosition")
    
    return end_y

def _commit_point_to_path(points_list_lanes, sow_flags_list, new_lane_point, sow_flag_requested, context=""):
    # new_lane_point is (lane_x, lane_y)
    if not points_list_lanes:
        points_list_lanes.append(new_lane_point)
        return
    if points_list_lanes[-1] == new_lane_point:
        return

    previous_lane_point = points_list_lanes[-1]
    current_segment = frozenset({previous_lane_point, new_lane_point})
    actual_sow_flag_for_this_segment = False

    if sow_flag_requested:
        # FIXED: Handle boundary segments specially to ensure visualization
        if "Boundary" in context:
            # Always allow boundary segments to be marked as sown for proper visualization
            actual_sow_flag_for_this_segment = True
        elif current_segment not in SOWN_SEGMENTS_LOG:
            actual_sow_flag_for_this_segment = True
            SOWN_SEGMENTS_LOG.add(current_segment)
        
    points_list_lanes.append(new_lane_point)
    sow_flags_list.append(actual_sow_flag_for_this_segment)


def _add_headland_segment_custom_exit(current_lane_x, current_lane_y, 
                                     target_lane_x, target_lane_y, 
                                     exit_point_lanes, 
                                     _points_list_lanes, _sow_flags_list, 
                                     segment_label="",
                                     is_designated_unsown_positioning_leg=False,
                                     gap_size=1):
    ex_lane_x, ey_lane_y = exit_point_lanes
    if (current_lane_x, current_lane_y) == exit_point_lanes:
        _commit_point_to_path(_points_list_lanes, _sow_flags_list, (target_lane_x, target_lane_y), False, f"AHLCE {segment_label} Case 1")
        return target_lane_x, target_lane_y

    on_segment_path = False
    if current_lane_x == target_lane_x == ex_lane_x and min(current_lane_y, target_lane_y) <= ey_lane_y <= max(current_lane_y, target_lane_y):
        on_segment_path = True
    elif current_lane_y == target_lane_y == ey_lane_y and min(current_lane_x, target_lane_x) <= ex_lane_x <= max(current_lane_x, target_lane_x):
        on_segment_path = True

    if on_segment_path:
        # Calculate stop point before exit to leave gap
        if current_lane_x == target_lane_x == ex_lane_x:  # Vertical movement
            if current_lane_y < ey_lane_y:  # Moving upward
                sow_stop_y = max(current_lane_y, ey_lane_y - gap_size)
            else:  # Moving downward
                sow_stop_y = min(current_lane_y, ey_lane_y + gap_size)
            
            # Sow until stop point
            if sow_stop_y != current_lane_y:
                _commit_point_to_path(_points_list_lanes, _sow_flags_list, (ex_lane_x, sow_stop_y), True, f"AHLCE {segment_label} Case 2 SowToGap")
            
            # Unsown movement to exit
            _commit_point_to_path(_points_list_lanes, _sow_flags_list, (ex_lane_x, ey_lane_y), False, f"AHLCE {segment_label} Case 2 ToExit")
            
            # Continue unsown to target if needed
            if (ex_lane_x, ey_lane_y) != (target_lane_x, target_lane_y):
                _commit_point_to_path(_points_list_lanes, _sow_flags_list, (target_lane_x, target_lane_y), False, f"AHLCE {segment_label} Case 2 PastExit")
                
        elif current_lane_y == target_lane_y == ey_lane_y:  # Horizontal movement
            if current_lane_x < ex_lane_x:  # Moving rightward
                sow_stop_x = max(current_lane_x, ex_lane_x - gap_size)
            else:  # Moving leftward
                sow_stop_x = min(current_lane_x, ex_lane_x + gap_size)
            
            # Sow until stop point
            if sow_stop_x != current_lane_x:
                _commit_point_to_path(_points_list_lanes, _sow_flags_list, (sow_stop_x, ey_lane_y), True, f"AHLCE {segment_label} Case 2 SowToGap")
            
            # Unsown movement to exit
            _commit_point_to_path(_points_list_lanes, _sow_flags_list, (ex_lane_x, ey_lane_y), False, f"AHLCE {segment_label} Case 2 ToExit")
            
            # Continue unsown to target if needed
            if (ex_lane_x, ey_lane_y) != (target_lane_x, target_lane_y):
                _commit_point_to_path(_points_list_lanes, _sow_flags_list, (target_lane_x, target_lane_y), False, f"AHLCE {segment_label} Case 2 PastExit")
    else: 
        sow_request = not is_designated_unsown_positioning_leg
        _commit_point_to_path(_points_list_lanes, _sow_flags_list, (target_lane_x, target_lane_y), sow_request, f"AHLCE {segment_label} Case 3")
            
    return target_lane_x, target_lane_y

def get_user_choice_corner_lanes(max_lx_idx, max_ly_idx):
    print(f"\nChoose exit corner (0-indexed lanes: X up to {max_lx_idx}, Y up to {max_ly_idx}):")
    print(f"1. Top-Left Lane (0, {max_ly_idx})")
    print(f"2. Top-Right Lane ({max_lx_idx}, {max_ly_idx})")
    print(f"3. Bottom-Left Lane (0, 0)")
    print(f"4. Bottom-Right Lane ({max_lx_idx}, 0)")
    while True:
        try:
            choice = int(input("Enter choice (1-4): "))
            if choice == 1: return (0, max_ly_idx)
            if choice == 2: return (max_lx_idx, max_ly_idx)
            if choice == 3: return (0, 0)
            if choice == 4: return (max_lx_idx, 0)
        except ValueError: print("Invalid input.")

def get_user_defined_exit_lanes(max_lx_idx, max_ly_idx):
    print(f"\nDefine custom exit lane (X: 0-{max_lx_idx}, Y: 0-{max_ly_idx}):")
    print(f"1. Top boundary (Y lane = {max_ly_idx})")
    print(f"2. Bottom boundary (Y lane = 0)")
    print(f"3. Left boundary (X lane = 0)")
    print(f"4. Right boundary (X lane = {max_lx_idx})")
    while True:
        try:
            b_choice = int(input("Choose boundary for exit (1-4): "))
            if 1 <= b_choice <= 4: break
        except ValueError: print("Invalid input.")
    
    ex_l, ey_l = -1, -1
    while True:
        try:
            if b_choice == 1: ey_l = max_ly_idx; ex_l = int(input(f"Enter X-lane (0-{max_lx_idx}): ")); assert 0 <= ex_l <= max_lx_idx; break
            if b_choice == 2: ey_l = 0; ex_l = int(input(f"Enter X-lane (0-{max_lx_idx}): ")); assert 0 <= ex_l <= max_lx_idx; break
            if b_choice == 3: ex_l = 0; ey_l = int(input(f"Enter Y-lane (0-{max_ly_idx}): ")); assert 0 <= ey_l <= max_ly_idx; break
            if b_choice == 4: ex_l = max_lx_idx; ey_l = int(input(f"Enter Y-lane (0-{max_ly_idx}): ")); assert 0 <= ey_l <= max_ly_idx; break
        except (ValueError, AssertionError): print("Invalid lane index.")
    return (ex_l, ey_l)

def _commit_partial_vertical_sweep(points_list, sow_flags_list, curr_x, start_y, end_y, gap_size=1):
    """
    Creates a partial vertical sweep with gaps at both ends.
    
    Args:
        points_list: List of lane points
        sow_flags_list: List of sowing flags
        curr_x: Current X lane position
        start_y: Starting Y lane position
        end_y: Ending Y lane position  
        gap_size: Size of gap to leave at each end (in lane units)
    """
    if start_y == end_y:
        return end_y
    
    # Determine direction and calculate total distance
    direction = 1 if end_y > start_y else -1
    total_distance = abs(end_y - start_y)
    
    # If total distance is too small for meaningful gaps, do full sown movement
    if total_distance <= 2 * gap_size:
        _commit_point_to_path(points_list, sow_flags_list, (curr_x, end_y), True, "PartialSweep_FullSown")
        return end_y
    
    # Calculate sowing start and end positions
    # Leave gap_size at the beginning and end
    sow_start_y = start_y + (gap_size * direction)
    sow_end_y = end_y - (gap_size * direction)
    
    # Phase 1: Unsown movement to sowing start position (creating initial gap)
    if sow_start_y != start_y:
        _commit_point_to_path(points_list, sow_flags_list, (curr_x, sow_start_y), False, "PartialSweep_Phase1_Gap")
    
    # Phase 2: Sown movement (main productive sweep)
    _commit_point_to_path(points_list, sow_flags_list, (curr_x, sow_end_y), True, "PartialSweep_Phase2_Sown")
    
    # Phase 3: Unsown movement to final position (creating final gap)
    if sow_end_y != end_y:
        _commit_point_to_path(points_list, sow_flags_list, (curr_x, end_y), False, "PartialSweep_Phase3_Gap")
    
    return end_y

# --- Path Generation (Operates in 0-indexed Lane Numbers) ---
# Updated helper function to properly handle gap_size parameter
def _add_headland_segment_custom_exit_with_gaps(current_lane_x, current_lane_y, 
                                     target_lane_x, target_lane_y, 
                                     exit_point_lanes, 
                                     _points_list_lanes, _sow_flags_list, 
                                     segment_label="",
                                     is_designated_unsown_positioning_leg=False,
                                     gap_size=1):
    """
    Modified version of _add_headland_segment_custom_exit that always considers gap_size
    for consistent behavior with inner sweeps.
    """
    ex_lane_x, ey_lane_y = exit_point_lanes
    if (current_lane_x, current_lane_y) == exit_point_lanes:
        _commit_point_to_path(_points_list_lanes, _sow_flags_list, (target_lane_x, target_lane_y), False, f"AHLCE {segment_label} Case 1")
        return target_lane_x, target_lane_y

    on_segment_path = False
    if current_lane_x == target_lane_x == ex_lane_x and min(current_lane_y, target_lane_y) <= ey_lane_y <= max(current_lane_y, target_lane_y):
        on_segment_path = True
    elif current_lane_y == target_lane_y == ey_lane_y and min(current_lane_x, target_lane_x) <= ex_lane_x <= max(current_lane_x, target_lane_x):
        on_segment_path = True

    if on_segment_path:
        # Calculate stop point before exit to leave gap
        if current_lane_x == target_lane_x == ex_lane_x:  # Vertical movement
            if current_lane_y < ey_lane_y:  # Moving upward
                sow_stop_y = max(current_lane_y, ey_lane_y - gap_size)
            else:  # Moving downward
                sow_stop_y = min(current_lane_y, ey_lane_y + gap_size)
            
            # Sow until stop point
            if sow_stop_y != current_lane_y:
                _commit_point_to_path(_points_list_lanes, _sow_flags_list, (ex_lane_x, sow_stop_y), True, f"AHLCE {segment_label} Case 2 SowToGap")
            
            # Unsown movement to exit
            _commit_point_to_path(_points_list_lanes, _sow_flags_list, (ex_lane_x, ey_lane_y), False, f"AHLCE {segment_label} Case 2 ToExit")
            
            # Continue unsown to target if needed
            if (ex_lane_x, ey_lane_y) != (target_lane_x, target_lane_y):
                _commit_point_to_path(_points_list_lanes, _sow_flags_list, (target_lane_x, target_lane_y), False, f"AHLCE {segment_label} Case 2 PastExit")
                
        elif current_lane_y == target_lane_y == ey_lane_y:  # Horizontal movement
            if current_lane_x < ex_lane_x:  # Moving rightward
                sow_stop_x = max(current_lane_x, ex_lane_x - gap_size)
            else:  # Moving leftward
                sow_stop_x = min(current_lane_x, ex_lane_x + gap_size)
            
            # Sow until stop point
            if sow_stop_x != current_lane_x:
                _commit_point_to_path(_points_list_lanes, _sow_flags_list, (sow_stop_x, ey_lane_y), True, f"AHLCE {segment_label} Case 2 SowToGap")
            
            # Unsown movement to exit
            _commit_point_to_path(_points_list_lanes, _sow_flags_list, (ex_lane_x, ey_lane_y), False, f"AHLCE {segment_label} Case 2 ToExit")
            
            # Continue unsown to target if needed
            if (ex_lane_x, ey_lane_y) != (target_lane_x, target_lane_y):
                _commit_point_to_path(_points_list_lanes, _sow_flags_list, (target_lane_x, target_lane_y), False, f"AHLCE {segment_label} Case 2 PastExit")
    else: 
        # Exit not on this segment - use partial sweep logic for gap consistency when coming from inner sweeps
        if current_lane_x == target_lane_x and current_lane_y != target_lane_y:
            # This is a vertical movement - apply partial sweep logic for gap consistency
            if gap_size > 0:
                # Use partial vertical sweep logic
                direction = 1 if target_lane_y > current_lane_y else -1
                total_distance = abs(target_lane_y - current_lane_y)
                
                # If total distance is too small for meaningful gaps, do full sown movement
                if total_distance <= 2 * gap_size:
                    sow_request = not is_designated_unsown_positioning_leg
                    _commit_point_to_path(_points_list_lanes, _sow_flags_list, (target_lane_x, target_lane_y), sow_request, f"AHLCE {segment_label} Case 3 FullSown")
                else:
                    # Apply gap logic similar to _commit_partial_vertical_sweep
                    sow_start_y = current_lane_y + (gap_size * direction)
                    sow_end_y = target_lane_y - (gap_size * direction)
                    
                    # Phase 1: Unsown movement to sowing start position (creating initial gap)
                    if sow_start_y != current_lane_y:
                        _commit_point_to_path(_points_list_lanes, _sow_flags_list, (current_lane_x, sow_start_y), False, f"AHLCE {segment_label} Case 3 InitialGap")
                    
                    # Phase 2: Sown movement (main productive sweep)
                    if not is_designated_unsown_positioning_leg:
                        _commit_point_to_path(_points_list_lanes, _sow_flags_list, (current_lane_x, sow_end_y), True, f"AHLCE {segment_label} Case 3 SownMiddle")
                    else:
                        _commit_point_to_path(_points_list_lanes, _sow_flags_list, (current_lane_x, sow_end_y), False, f"AHLCE {segment_label} Case 3 UnsownMiddle")
                    
                    # Phase 3: Unsown movement to final position (creating final gap)
                    if sow_end_y != target_lane_y:
                        _commit_point_to_path(_points_list_lanes, _sow_flags_list, (target_lane_x, target_lane_y), False, f"AHLCE {segment_label} Case 3 FinalGap")
            else:
                # No gap_size specified, use original logic
                sow_request = not is_designated_unsown_positioning_leg
                _commit_point_to_path(_points_list_lanes, _sow_flags_list, (target_lane_x, target_lane_y), sow_request, f"AHLCE {segment_label} Case 3 NoGap")
        else:
            # Horizontal movement or other cases - use original logic
            sow_request = not is_designated_unsown_positioning_leg
            _commit_point_to_path(_points_list_lanes, _sow_flags_list, (target_lane_x, target_lane_y), sow_request, f"AHLCE {segment_label} Case 3 Other")
            
    return target_lane_x, target_lane_y

def determine_optimal_start_lane(n_inner_x_sweeps, max_lane_idx_x, exit_point_lanes, is_corner_exit):
    """
    Determines the optimal starting lane based on exit location for fastest completion.
    """
    if n_inner_x_sweeps <= 0:
        return 0 if exit_point_lanes[0] <= max_lane_idx_x / 2.0 else max_lane_idx_x
    
    first_inner_lane_x = 1
    last_inner_lane_x = max_lane_idx_x - 1
    ex_ln_x = exit_point_lanes[0]
    
    if is_corner_exit:
        # For corner exits, start from the side that leads to shorter headland path
        return first_inner_lane_x if ex_ln_x <= max_lane_idx_x / 2.0 else last_inner_lane_x
    else:
        # For custom exits, calculate optimal start based on total path distance
        min_total_distance = float('inf')
        optimal_start = first_inner_lane_x
        
        for potential_start in range(first_inner_lane_x, last_inner_lane_x + 1):
            # Estimate total distance for this starting position
            inner_sweep_distance = n_inner_x_sweeps * (max_lane_idx_x + 1)  # Rough estimate
            headland_distance = 2 * (max_lane_idx_x + max_lane_idx_x + 1)  # Perimeter
            exit_approach_distance = abs(potential_start - ex_ln_x)
            
            total_estimated_distance = inner_sweep_distance + headland_distance + exit_approach_distance
            
            if total_estimated_distance < min_total_distance:
                min_total_distance = total_estimated_distance
                optimal_start = potential_start
        
        return optimal_start

def generate_optimized_sweep_sequence(start_lane_x, n_inner_x_sweeps, max_lane_idx_x, exit_point_lanes, is_corner_exit):
    """
    Generates an optimized sequence of inner sweeps that minimizes total travel time to exit.
    """
    if n_inner_x_sweeps <= 0:
        return []
    
    first_inner_lane_x = 1
    last_inner_lane_x = max_lane_idx_x - 1
    all_inner_lanes = list(range(first_inner_lane_x, last_inner_lane_x + 1))
    
    if n_inner_x_sweeps == 1:
        return [start_lane_x]
    
    if is_corner_exit:
        # Standard boustrophedon for corner exits
        direction = -1 if start_lane_x == last_inner_lane_x else 1
        end_range = (first_inner_lane_x - 1) if direction == -1 else (last_inner_lane_x + 1)
        return list(range(start_lane_x, end_range, direction))[:n_inner_x_sweeps]
    else:
        # Optimized sequence for custom exits
        ex_ln_x = exit_point_lanes[0]
        sequence = [start_lane_x]
        remaining_lanes = [x for x in all_inner_lanes if x != start_lane_x]
        
        # Sort remaining lanes by distance to exit (closer lanes first)
        remaining_lanes.sort(key=lambda x: abs(x - ex_ln_x))
        
        # Add lanes in alternating pattern while respecting exit proximity
        current_pos = start_lane_x
        for _ in range(min(n_inner_x_sweeps - 1, len(remaining_lanes))):
            # Find the next best lane considering both distance to exit and boustrophedon efficiency
            best_lane = None
            best_score = float('inf')
            
            for lane in remaining_lanes:
                if lane in sequence:
                    continue
                
                # Score based on: distance to exit + transition cost from current position
                exit_distance_score = abs(lane - ex_ln_x)
                transition_cost = abs(lane - current_pos)
                total_score = exit_distance_score + (transition_cost * 0.5)  # Weight transition cost less
                
                if total_score < best_score:
                    best_score = total_score
                    best_lane = lane
            
            if best_lane is not None:
                sequence.append(best_lane)
                current_pos = best_lane
        
        return sequence[:n_inner_x_sweeps]



def generate_fixed_path_optimized(n_inner_x_sweeps, max_lane_idx_x, max_lane_idx_y, exit_point_lanes, is_corner_exit, gap_size=1):
    """
    COMPLETELY FIXED: Optimized path generation that works for ANY farm dimensions.
    """
    global SOWN_SEGMENTS_LOG
    SOWN_SEGMENTS_LOG.clear() 

    _points_lanes = []
    _sow_flags = []
    ex_ln_x, ex_ln_y = exit_point_lanes
    
    def _commit(new_ln_pt, sow_req, ctx=""):
        """FIXED: Always commit points and flags properly"""
        if not _points_lanes:
            _points_lanes.append(new_ln_pt)
            return
        if _points_lanes[-1] == new_ln_pt:
            return
        
        _points_lanes.append(new_ln_pt)
        _sow_flags.append(sow_req)
        print(f"  ✅ Committed: {_points_lanes[-2] if len(_points_lanes) > 1 else 'START'} → {new_ln_pt} | Sow: {sow_req} | Context: {ctx}")

    # FIXED: Calculate inner lane boundaries properly for ANY dimensions
    first_inner_lane_x = 1 if max_lane_idx_x >= 2 else 0
    last_inner_lane_x = max_lane_idx_x - 1 if max_lane_idx_x >= 2 else max_lane_idx_x
    
    print(f"🔍 DEBUG: Farm grid {max_lane_idx_x+1}x{max_lane_idx_y+1} lanes")
    print(f"🔍 DEBUG: Inner lanes X: {first_inner_lane_x} to {last_inner_lane_x}")
    print(f"🔍 DEBUG: Requested inner sweeps: {n_inner_x_sweeps}")
    print(f"🔍 DEBUG: Exit at lane: ({ex_ln_x}, {ex_ln_y})")
    
    # FIXED: Smart start position logic for ANY dimensions
    if n_inner_x_sweeps > 0 and first_inner_lane_x <= last_inner_lane_x:
        # Determine optimal start based on exit position
        if ex_ln_x == 0:  # Exit on LEFT boundary
            start_sweep_lane_x = last_inner_lane_x
            print(f"🎯 Exit on LEFT - starting from rightmost inner lane x={start_sweep_lane_x}")
        elif ex_ln_x == max_lane_idx_x:  # Exit on RIGHT boundary
            start_sweep_lane_x = first_inner_lane_x
            print(f"🎯 Exit on RIGHT - starting from leftmost inner lane x={start_sweep_lane_x}")
        else:  # Exit on TOP/BOTTOM boundary
            # Choose farthest inner lane from exit X position
            if abs(ex_ln_x - first_inner_lane_x) > abs(ex_ln_x - last_inner_lane_x):
                start_sweep_lane_x = first_inner_lane_x
            else:
                start_sweep_lane_x = last_inner_lane_x
            print(f"🎯 Exit on TOP/BOTTOM - starting from farthest inner lane x={start_sweep_lane_x}")
    else:
        # No inner sweeps or invalid inner lane range
        start_sweep_lane_x = 0 if ex_ln_x > max_lane_idx_x / 2.0 else max_lane_idx_x
        print(f"🎯 No inner sweeps - starting from boundary lane x={start_sweep_lane_x}")

    # FIXED: Determine start_lane_y based on exit position for ANY dimensions
    if ex_ln_y == 0:  # Exit on BOTTOM boundary
        b_start_lane_y = max_lane_idx_y  # Start from TOP
        print(f"🎯 Exit on BOTTOM - starting from TOP y={b_start_lane_y}")
    elif ex_ln_y == max_lane_idx_y:  # Exit on TOP boundary  
        b_start_lane_y = 0  # Start from BOTTOM
        print(f"🎯 Exit on TOP - starting from BOTTOM y={b_start_lane_y}")
    else:
        # For side exits, choose based on distance
        if abs(ex_ln_y - max_lane_idx_y) < abs(ex_ln_y - 0):
            b_start_lane_y = 0  # Start from bottom if exit closer to top
        else:
            b_start_lane_y = max_lane_idx_y  # Start from top if exit closer to bottom
        print(f"🎯 Exit on SIDE - starting from y={b_start_lane_y}")
    
    # FIXED: Generate sweep sequence - WORKS FOR ANY DIMENSIONS
    lanes_to_sweep_x = []
    if n_inner_x_sweeps > 0 and first_inner_lane_x <= last_inner_lane_x:
        all_inner_lanes_x = list(range(first_inner_lane_x, last_inner_lane_x + 1))
        available_inner_lanes = min(len(all_inner_lanes_x), n_inner_x_sweeps)
        
        print(f"🔍 All inner lanes: {all_inner_lanes_x}")
        print(f"🔍 Available inner lanes: {available_inner_lanes}")
        
        if available_inner_lanes == 1:
            lanes_to_sweep_x = [start_sweep_lane_x]
        elif available_inner_lanes > 1:
            # Start from chosen lane, then fill systematically
            lanes_to_sweep_x = [start_sweep_lane_x]
            remaining_lanes = [x for x in all_inner_lanes_x if x != start_sweep_lane_x]
            
            # Add remaining lanes in order (closest to start first for efficiency)
            remaining_lanes.sort(key=lambda x: abs(x - start_sweep_lane_x))
            lanes_to_sweep_x.extend(remaining_lanes[:available_inner_lanes-1])
        
        print(f"📋 Final sweep sequence: {lanes_to_sweep_x}")

    # FIXED: Initialize position properly
    initial_pos_lane_x = start_sweep_lane_x if lanes_to_sweep_x else (0 if ex_ln_x > max_lane_idx_x/2 else max_lane_idx_x)
    _points_lanes.append((initial_pos_lane_x, b_start_lane_y))
    curr_ln_x, curr_ln_y = initial_pos_lane_x, b_start_lane_y
    print(f"🚀 Starting at lane ({curr_ln_x}, {curr_ln_y})")

    # FIXED: Execute inner sweeps with proper logic for ANY dimensions
    for i, sweep_ln_x in enumerate(lanes_to_sweep_x):
        print(f"\n🔄 Processing VRow {i+1} at lane x={sweep_ln_x}")
        
        # Move to sweep lane if needed
        if curr_ln_x != sweep_ln_x:
            _commit((sweep_ln_x, curr_ln_y), False, f"MoveToVRow{i+1}")
            curr_ln_x = sweep_ln_x
        
        # Determine target Y for this sweep
        target_ln_y = 0 if curr_ln_y == max_lane_idx_y else max_lane_idx_y
        
        # Check if this is the last VRow and we need special handling
        is_last_vrow = (i == len(lanes_to_sweep_x) - 1)
        
        if is_last_vrow and (ex_ln_y == 0 or ex_ln_y == max_lane_idx_y):
            print(f"🎯 Last VRow - using smart transition to avoid exit area")
            
            # Smart transition logic for ANY dimensions
            direction = 1 if target_ln_y > curr_ln_y else -1
            total_distance = abs(target_ln_y - curr_ln_y)
            
            if total_distance <= gap_size * 2:
                _commit((curr_ln_x, target_ln_y), False, "SmartTransition_TooShort")
                curr_ln_y = target_ln_y
            else:
                # Phase 1: Small initial gap
                gap_end_y = curr_ln_y + (gap_size * direction)
                gap_end_y = max(0, min(max_lane_idx_y, gap_end_y))
                if gap_end_y != curr_ln_y:
                    _commit((curr_ln_x, gap_end_y), False, "SmartTransition_InitialGap")
                    curr_ln_y = gap_end_y
                
                # Phase 2: Sow most of the distance
                sow_distance = int(total_distance * 0.7)  # Sow 70% of distance
                sow_end_y = curr_ln_y + (sow_distance * direction)
                sow_end_y = max(0, min(max_lane_idx_y, sow_end_y))
                if sow_end_y != curr_ln_y:
                    _commit((curr_ln_x, sow_end_y), True, "SmartTransition_MainSow")
                    curr_ln_y = sow_end_y
                
                # Phase 3: Final positioning
                if curr_ln_y != target_ln_y:
                    _commit((curr_ln_x, target_ln_y), False, "SmartTransition_FinalPosition")
                    curr_ln_y = target_ln_y
        else:
            # Regular inner sweep with proper gap handling
            print(f"🌱 Regular VRow sweep from y={curr_ln_y} to y={target_ln_y}")
            
            direction = 1 if target_ln_y > curr_ln_y else -1
            total_distance = abs(target_ln_y - curr_ln_y)
            
            if total_distance <= 2 * gap_size:
                # Too short for gaps, just sow the whole thing
                _commit((curr_ln_x, target_ln_y), True, f"VRow{i+1}_FullSow")
                curr_ln_y = target_ln_y
            else:
                # Phase 1: Initial gap
                gap_start_y = curr_ln_y + (gap_size * direction)
                gap_start_y = max(0, min(max_lane_idx_y, gap_start_y))
                if gap_start_y != curr_ln_y:
                    _commit((curr_ln_x, gap_start_y), False, f"VRow{i+1}_InitialGap")
                    curr_ln_y = gap_start_y
                
                # Phase 2: Main sowing
                sow_end_y = target_ln_y - (gap_size * direction)
                sow_end_y = max(0, min(max_lane_idx_y, sow_end_y))
                if sow_end_y != curr_ln_y:
                    _commit((curr_ln_x, sow_end_y), True, f"VRow{i+1}_MainSow")
                    curr_ln_y = sow_end_y
                
                # Phase 3: Final gap
                if curr_ln_y != target_ln_y:
                    _commit((curr_ln_x, target_ln_y), False, f"VRow{i+1}_FinalGap")
                    curr_ln_y = target_ln_y
        
        print(f"✅ VRow {i+1} completed at lane x={sweep_ln_x}")

        # FIXED: Horizontal transition logic for ANY dimensions
        if i < len(lanes_to_sweep_x) - 1: 
            next_sweep_ln_x = lanes_to_sweep_x[i+1]
            if curr_ln_x != next_sweep_ln_x:
                _commit((next_sweep_ln_x, curr_ln_y), False, f"TransitionToVRow{i+2}")
                curr_ln_x = next_sweep_ln_x
        else:
            # Last VRow - determine turning direction
            if ex_ln_y == 0 or ex_ln_y == max_lane_idx_y:  # Exit on top/bottom
                if ex_ln_x < curr_ln_x:
                    turn_target_x = max(0, curr_ln_x - 1)
                    print(f"🔄 Last VRow turning LEFT towards exit")
                elif ex_ln_x > curr_ln_x:
                    turn_target_x = min(max_lane_idx_x, curr_ln_x + 1)
                    print(f"🔄 Last VRow turning RIGHT towards exit")
                else:
                    # Aligned with exit - choose direction based on boundary proximity
                    if curr_ln_x <= max_lane_idx_x / 2:
                        turn_target_x = max(0, curr_ln_x - 1)
                    else:
                        turn_target_x = min(max_lane_idx_x, curr_ln_x + 1)
                    print(f"🔄 Last VRow aligned - turning towards boundary")
                
                if turn_target_x != curr_ln_x:
                    _commit((turn_target_x, curr_ln_y), False, f"TurnTowardsExit")
                    curr_ln_x = turn_target_x

    # FIXED: BOUNDARY COVERAGE - WORKS FOR ANY DIMENSIONS
    print(f"\n🔄 Starting boundary coverage from ({curr_ln_x}, {curr_ln_y})")
    
    # Systematic boundary coverage that works for any farm size
    if ex_ln_y == 0 or ex_ln_y == max_lane_idx_y:  # Exit on top/bottom boundary
        print(f"🔄 Exit on horizontal boundary at ({ex_ln_x}, {ex_ln_y})")
        
        # Strategy: Complete one vertical boundary, then horizontal sweep to other boundary
        if curr_ln_x < ex_ln_x:
            # Current position LEFT of exit - go to left boundary first
            print(f"🎯 Going to LEFT boundary first")
            
            # Go to left boundary
            if curr_ln_x != 0:
                _commit((0, curr_ln_y), True, "BoundaryToLeft")
                curr_ln_x = 0
            
            # Complete vertical sweep on left boundary
            opposite_y = 0 if curr_ln_y == max_lane_idx_y else max_lane_idx_y
            if curr_ln_y != opposite_y:
                _commit((0, opposite_y), True, "BoundaryLeftVertical")
                curr_ln_y = opposite_y
            
            # Horizontal sweep towards exit boundary
            if curr_ln_y == ex_ln_y:
                # On exit boundary - sow up to near exit, then position to exit, then continue
                if ex_ln_x > 0:
                    _commit((ex_ln_x - 1, curr_ln_y), True, "BoundaryToNearExit")
                    curr_ln_x = ex_ln_x - 1
                
                _commit((ex_ln_x, curr_ln_y), False, "BoundaryToExit")
                curr_ln_x = ex_ln_x
                
                if curr_ln_x != max_lane_idx_x:
                    _commit((max_lane_idx_x, curr_ln_y), True, "BoundaryToRight")
                    curr_ln_x = max_lane_idx_x
            else:
                # Not on exit boundary - normal horizontal sweep
                _commit((max_lane_idx_x, curr_ln_y), True, "BoundaryHorizontalToRight")
                curr_ln_x = max_lane_idx_x
            
            # Complete right boundary to exit level
            if curr_ln_y != ex_ln_y:
                _commit((max_lane_idx_x, ex_ln_y), True, "BoundaryRightToExitLevel")
                curr_ln_y = ex_ln_y
            
            # Final approach to exit
            if curr_ln_x != ex_ln_x:
                _commit((ex_ln_x, curr_ln_y), True, "BoundaryFinalToExit")
                curr_ln_x = ex_ln_x
                
        elif curr_ln_x > ex_ln_x:
            # Current position RIGHT of exit - go to right boundary first
            print(f"🎯 Going to RIGHT boundary first")
            
            # Go to right boundary
            if curr_ln_x != max_lane_idx_x:
                _commit((max_lane_idx_x, curr_ln_y), True, "BoundaryToRight")
                curr_ln_x = max_lane_idx_x
            
            # Complete vertical sweep on right boundary
            opposite_y = 0 if curr_ln_y == max_lane_idx_y else max_lane_idx_y
            if curr_ln_y != opposite_y:
                _commit((max_lane_idx_x, opposite_y), True, "BoundaryRightVertical")
                curr_ln_y = opposite_y
            
            # Horizontal sweep towards exit boundary
            if curr_ln_y == ex_ln_y:
                # On exit boundary - sow up to near exit, then position to exit, then continue
                if ex_ln_x < max_lane_idx_x:
                    _commit((ex_ln_x + 1, curr_ln_y), True, "BoundaryToNearExit")
                    curr_ln_x = ex_ln_x + 1
                
                _commit((ex_ln_x, curr_ln_y), False, "BoundaryToExit")
                curr_ln_x = ex_ln_x
                
                if curr_ln_x != 0:
                    _commit((0, curr_ln_y), True, "BoundaryToLeft")
                    curr_ln_x = 0
            else:
                # Not on exit boundary - normal horizontal sweep
                _commit((0, curr_ln_y), True, "BoundaryHorizontalToLeft")
                curr_ln_x = 0
            
            # Complete left boundary to exit level
            if curr_ln_y != ex_ln_y:
                _commit((0, ex_ln_y), True, "BoundaryLeftToExitLevel")
                curr_ln_y = ex_ln_y
            
            # Final approach to exit
            if curr_ln_x != ex_ln_x:
                _commit((ex_ln_x, curr_ln_y), True, "BoundaryFinalToExit")
                curr_ln_x = ex_ln_x
        
        else:
            # curr_ln_x == ex_ln_x: Directly aligned with exit
            print(f"🎯 Directly aligned with exit - choosing optimal boundary")
            
            if curr_ln_x <= max_lane_idx_x / 2:
                # Closer to left boundary
                if curr_ln_x != 0:
                    _commit((0, curr_ln_y), True, "BoundaryToLeft")
                    curr_ln_x = 0
                
                opposite_y = 0 if curr_ln_y == max_lane_idx_y else max_lane_idx_y
                if curr_ln_y != opposite_y:
                    _commit((0, opposite_y), True, "BoundaryLeftVertical")
                    curr_ln_y = opposite_y
                
                _commit((max_lane_idx_x, curr_ln_y), True, "BoundaryHorizontalToRight")
                curr_ln_x = max_lane_idx_x
                
                if curr_ln_y != ex_ln_y:
                    _commit((max_lane_idx_x, ex_ln_y), True, "BoundaryRightToExitLevel")
                    curr_ln_y = ex_ln_y
                
                if curr_ln_x != ex_ln_x:
                    _commit((ex_ln_x, curr_ln_y), True, "BoundaryFinalToExit")
                    curr_ln_x = ex_ln_x
            else:
                # Closer to right boundary
                if curr_ln_x != max_lane_idx_x:
                    _commit((max_lane_idx_x, curr_ln_y), True, "BoundaryToRight")
                    curr_ln_x = max_lane_idx_x
                
                opposite_y = 0 if curr_ln_y == max_lane_idx_y else max_lane_idx_y
                if curr_ln_y != opposite_y:
                    _commit((max_lane_idx_x, opposite_y), True, "BoundaryRightVertical")
                    curr_ln_y = opposite_y
                
                _commit((0, curr_ln_y), True, "BoundaryHorizontalToLeft")
                curr_ln_x = 0
                
                if curr_ln_y != ex_ln_y:
                    _commit((0, ex_ln_y), True, "BoundaryLeftToExitLevel")
                    curr_ln_y = ex_ln_y
                
                if curr_ln_x != ex_ln_x:
                    _commit((ex_ln_x, curr_ln_y), True, "BoundaryFinalToExit")
                    curr_ln_x = ex_ln_x
                
    elif ex_ln_x == 0 or ex_ln_x == max_lane_idx_x:  # Exit on left/right boundary
        print(f"🔄 Exit on vertical boundary at ({ex_ln_x}, {ex_ln_y})")
        
        # Strategy: Go to opposite vertical boundary, complete it, then sweep to exit boundary
        opposite_boundary_x = max_lane_idx_x if ex_ln_x == 0 else 0
        
        # Move to opposite boundary
        if curr_ln_x != opposite_boundary_x:
            _commit((opposite_boundary_x, curr_ln_y), True, f"BoundaryToOpposite_x{opposite_boundary_x}")
            curr_ln_x = opposite_boundary_x
        
        # Complete vertical sweep on opposite boundary
        target_ln_y = 0 if curr_ln_y == max_lane_idx_y else max_lane_idx_y
        if curr_ln_y != target_ln_y:
            _commit((curr_ln_x, target_ln_y), True, "BoundaryOppositeVertical")
            curr_ln_y = target_ln_y
        
        # Horizontal sweep to exit boundary
        if curr_ln_x != ex_ln_x:
            _commit((ex_ln_x, curr_ln_y), True, f"BoundaryHorizontalToExit_x{ex_ln_x}")
            curr_ln_x = ex_ln_x
        
        # Complete exit boundary (avoiding exit until final approach)
        opposite_y = 0 if curr_ln_y == max_lane_idx_y else max_lane_idx_y
        if curr_ln_y != opposite_y:
            if opposite_y == ex_ln_y:
                # Approaching exit level - sow up to near exit
                near_exit_y = ex_ln_y - 1 if ex_ln_y > 0 else ex_ln_y + 1
                near_exit_y = max(0, min(max_lane_idx_y, near_exit_y))
                
                if near_exit_y != curr_ln_y:
                    _commit((curr_ln_x, near_exit_y), True, "BoundaryToNearExitY")
                    curr_ln_y = near_exit_y
                
                _commit((curr_ln_x, ex_ln_y), False, "BoundaryToExitY")
                curr_ln_y = ex_ln_y
            else:
                # Not approaching exit level - normal sweep
                _commit((curr_ln_x, opposite_y), True, "BoundaryExitVertical")
                curr_ln_y = opposite_y
        
        # Final positioning to exact exit
        if curr_ln_y != ex_ln_y:
            _commit((curr_ln_x, ex_ln_y), True, "BoundaryFinalToExitY")
            curr_ln_y = ex_ln_y
    
    # FIXED: Ensure we end exactly at exit point
    if not _points_lanes or _points_lanes[-1] != exit_point_lanes:
        _commit(exit_point_lanes, False, "FinalExitPosition")
        print(f"🎯 Final positioning at exact exit: {exit_point_lanes}")
    
    print(f"\n✅ Path generation completed!")
    print(f"📊 Total waypoints: {len(_points_lanes)}")
    print(f"📊 Total segments: {len(_sow_flags)}")
    print(f"📊 Sown segments: {sum(_sow_flags)}")
    print(f"📊 Unsown segments: {len(_sow_flags) - sum(_sow_flags)}")
    
    # Debug output - show first few and last few points
    print(f"\n🔍 Path preview:")
    for i in range(min(5, len(_points_lanes))):
        sow_status = "SOWN" if i < len(_sow_flags) and _sow_flags[i] else "UNSOWN"
        print(f"  Point {i}: {_points_lanes[i]} → {_points_lanes[i+1] if i+1 < len(_points_lanes) else 'END'} ({sow_status})")
    
    if len(_points_lanes) > 10:
        print(f"  ... ({len(_points_lanes)-10} points in between) ...")
        for i in range(max(5, len(_points_lanes)-5), len(_points_lanes)):
            if i < len(_points_lanes):
                sow_status = "SOWN" if i-1 < len(_sow_flags) and i-1 >= 0 and _sow_flags[i-1] else "UNSOWN"
                prev_point = _points_lanes[i-1] if i > 0 else "START"
                print(f"  Point {i}: {prev_point} → {_points_lanes[i]} ({sow_status})")
    
    return {'points_lanes': _points_lanes, 'sow_flags': _sow_flags}




def analyze_optimized_path_sequence(path_lanes_list, n_inner_x_sweeps_val, max_lx_idx_val, max_ly_idx_val, sow_flags_list, exit_lanes):
    """
    Enhanced path analysis that recognizes optimization patterns and provides better labeling.
    """
    if not path_lanes_list or len(path_lanes_list) < 2: 
        return []
    
    row_sequence = []
    v_counter = 1
    h_counter = 1
    labeled_v_lanes = set()
    labeled_h_lanes = set()
    
    first_inner_lx = 1
    last_inner_lx = max_lx_idx_val - 1
    ex_ln_x, ex_ln_y = exit_lanes

    for i in range(len(path_lanes_list) - 1):
        lx1, ly1 = path_lanes_list[i]
        lx2, ly2 = path_lanes_list[i+1]
        is_sown = i < len(sow_flags_list) and sow_flags_list[i]
        label = ""
        mov_type = "Other"
        
        # Enhanced labeling for optimized paths
        if lx1 == lx2 and ly1 != ly2: 
            mov_type = "VRow_Path"
            if is_sown and first_inner_lx <= lx1 <= last_inner_lx and lx1 not in labeled_v_lanes:
                # Check if this is an optimized partial sweep
                segment_distance = abs(ly2 - ly1)
                max_possible_distance = max_ly_idx_val
                
                if segment_distance < max_possible_distance * 0.8:
                    label = f"VRow{v_counter}_Optimized"
                else:
                    label = f"VRow{v_counter}"
                
                labeled_v_lanes.add(lx1)
                v_counter += 1
            elif is_sown and (lx1 == 0 or lx1 == max_lx_idx_val):
                label = f"Boundary_V{lx1}_Optimized" if abs(ly2 - ly1) < max_ly_idx_val else f"Boundary_V{lx1}"
        
        elif ly1 == ly2 and lx1 != lx2: 
            mov_type = "HRow_Path"
            if is_sown and (ly1 == 0 or ly1 == max_ly_idx_val) and ly1 not in labeled_h_lanes:
                # Check if this is approaching exit
                exit_approach = abs(lx2 - ex_ln_x) < abs(lx1 - ex_ln_x)
                label = f"HRow{h_counter}_ExitApproach" if exit_approach else f"HRow{h_counter}"
                labeled_h_lanes.add(ly1)
                h_counter += 1
            elif is_sown: 
                label = f"H-Turn{i+1}_Optimized"
        
        # Special labeling for optimization segments
        if not is_sown and mov_type in ["VRow_Path", "HRow_Path"]:
            if abs(lx2 - ex_ln_x) + abs(ly2 - ex_ln_y) < abs(lx1 - ex_ln_x) + abs(ly1 - ex_ln_y):
                label = f"ExitPositioning_{i+1}"
            else:
                label = f"Transition_{i+1}"
                    
        row_sequence.append({
            'segment_path_index': i, 
            'movement_type': mov_type, 
            'from_pos_lanes': (lx1, ly1), 
            'to_pos_lanes': (lx2, ly2),
            'label': label, 
            'is_sown': is_sown,
            'optimization_type': 'exit_approach' if 'ExitApproach' in label else 'standard'
        })
    
    return row_sequence


def get_movement_analysis(path_lanes_list, seg_idx, max_lx_idx_val, max_ly_idx_val, analyzed_row_seq, sow_flags_all, rover_width_m_val, rover_length_m_val):
    if seg_idx >= len(path_lanes_list) - 1 or seg_idx >= len(sow_flags_all): return None 
    
    lx1, ly1 = path_lanes_list[seg_idx]; lx2, ly2 = path_lanes_list[seg_idx+1]

    # FIXED: Use correct dimensions for lane center calculation - GENERIC
    from_pos_m_center = ((lx1 + 0.5) * rover_width_m_val, (ly1 + 0.5) * rover_length_m_val)
    to_pos_m_center = ((lx2 + 0.5) * rover_width_m_val, (ly2 + 0.5) * rover_length_m_val)

    # FIXED: Use correct dimensions for distance calculation - GENERIC
    distance_m = abs(lx2 - lx1) * rover_width_m_val + abs(ly2 - ly1) * rover_length_m_val
    
    row_info = next((rs for rs in analyzed_row_seq if rs['segment_path_index'] == seg_idx), None)
    label = row_info['label'] if row_info and row_info['label'] else f"Segment{seg_idx + 1}" 
    
    analysis = {'from_pos_m': from_pos_m_center, 'to_pos_m': to_pos_m_center,
                'from_row_y_coord_m': from_pos_m_center[1], 'to_row_y_coord_m': to_pos_m_center[1],
                'distance_m': distance_m, 'direction': '', 'action': '', 
                'farming_type': '', 'status': '', 'row_sequence_label': label}

    if lx2 > lx1: analysis['direction'] = 'EAST'
    elif lx2 < lx1: analysis['direction'] = 'WEST'
    elif ly2 > ly1: analysis['direction'] = 'NORTH'
    elif ly2 < ly1: analysis['direction'] = 'SOUTH'
    
    is_sown = sow_flags_all[seg_idx]
    first_inner_lx = 1
    last_inner_lx = max_lx_idx_val - 1

    if not is_sown:
        analysis['farming_type'] = 'NONE'; analysis['action'] = 'NAVIGATION_UNSOWN'; analysis['status'] = 'TRAVERSING_NO_SOW'
    else:
        if lx1 == lx2 and ly1 != ly2: 
            analysis['status'] = 'SOWING_VERTICALLY'
            is_inner_v_sweep = (first_inner_lx <= lx1 <= last_inner_lx)
            analysis['action'] = 'INNER_VERTICAL_FARMING' if is_inner_v_sweep else 'BOUNDARY_VERTICAL_FARMING'
            analysis['farming_type'] = 'CROP_PLANTING_V' if is_inner_v_sweep else 'PERIMETER_SOWING_V'
        elif ly1 == ly2 and lx1 != lx2: 
            analysis['status'] = 'SOWING_HORIZONTALLY'
            is_headland_h = (ly1 == 0 or ly1 == max_ly_idx_val)
            analysis['action'] = 'BOUNDARY_HORIZONTAL_FARMING' if is_headland_h else 'TRANSITION_SOWING_H' 
            analysis['farming_type'] = 'PERIMETER_SOWING_H' if is_headland_h else 'TRANSITION_SOWING_H'
        else: 
            analysis['action'] = 'DIAGONAL_SOWING_ERROR'; analysis['farming_type'] = 'ERROR_SOW'; analysis['status'] = 'SOWING_ERROR_PATH'
    return analysis


# --- Telemetry Logger ---
class LiveTelemetryLogger:
    def __init__(self, farm_w_m, farm_b_m, rover_lw_m, exit_info_str):
        self.farm_w_m = farm_w_m; self.farm_b_m = farm_b_m; self.rover_lw_m = rover_lw_m
        self.sown_v_segs = 0; self.sown_h_segs = 0
        self.total_dist_m = 0; self.total_sow_dist_m = 0
        self.start_time = datetime.now(); self.csv_filename = "navigation_log.csv"
        if not os.path.exists(self.csv_filename):
            try:
                with open(self.csv_filename, 'w', newline='', encoding='utf-8') as f:
                    w = csv.writer(f); w.writerow(["Timestamp", "Step", "Label", "From (m)", "To (m)", "FromY (m)", "ToY (m)", "Dir", "Action", "FarmType", "Status", "SegDist (m)", "TotalDist (m)", "SownDist (m)", "V_Sown_Segs", "H_Sown_Segs"])
                print(f"💾 CSV created: {self.csv_filename}")
            except IOError as e: print(f"❌ CSV Error: {e}")
        else: print(f"📝 Appending to: {self.csv_filename}")
        hdr = "🤖 FARM ROBOT TELEMETRY 🤖"; print(f"\n{hdr}\n{'='*len(hdr)}\n📊 Farm: {farm_w_m}x{farm_b_m}m, Rover Lane: {rover_lw_m}m\n🎯 Exit: {exit_info_str}\n⏰ Start: {self.start_time:%Y-%m-%d %H:%M:%S}\n{'='*len(hdr)}\n🔴 LIVE LOG:\n{'='*len(hdr)}")

    def log_movement(self, step, analysis, time_now): 
        if not analysis: return
        self.total_dist_m += analysis['distance_m']
        if analysis['farming_type'] != 'NONE': 
            self.total_sow_dist_m += analysis['distance_m']
            if 'VERTICAL' in analysis['action'] or '_V' in analysis['farming_type']: self.sown_v_segs += 1
            elif 'HORIZONTAL' in analysis['action'] or '_H' in analysis['farming_type']: self.sown_h_segs += 1
        elapsed = (time_now - self.start_time).total_seconds()
        display_label = analysis['row_sequence_label']
        from_pos_str = f"({analysis['from_pos_m'][0]:.1f}, {analysis['from_pos_m'][1]:.1f})"
        to_pos_str = f"({analysis['to_pos_m'][0]:.1f}, {analysis['to_pos_m'][1]:.1f})"
        print(f"\n⏱️ {time_now:%H:%M:%S.%f}"[:-3] + f" [S{step:02d}] (+{elapsed:.1f}s)\n🏷️ {display_label}\n📍 {from_pos_str} → {to_pos_str} (D:{analysis['distance_m']:.1f}m)\n🧭 Act: {analysis['action']} ({analysis['status']}) | Type: {analysis['farming_type']}\n📊 TD:{self.total_dist_m:.1f}m SD:{self.total_sow_dist_m:.1f}m VS:{self.sown_v_segs} HS:{self.sown_h_segs}\n{'-'*70}")
        row = [time_now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], step, display_label, str(analysis['from_pos_m']), str(analysis['to_pos_m']), f"{analysis['from_row_y_coord_m']:.1f}", f"{analysis['to_row_y_coord_m']:.1f}", analysis['direction'], analysis['action'], analysis['farming_type'], analysis['status'], f"{analysis['distance_m']:.1f}", f"{self.total_dist_m:.1f}", f"{self.total_sow_dist_m:.1f}", self.sown_v_segs, self.sown_h_segs]
        try:
            with open(self.csv_filename, 'a', newline='', encoding='utf-8') as f: csv.writer(f).writerow(row)
        except IOError as e: print(f"❌ CSV Write Err (S{step}): {e}")

    def finalize_mission(self, final_pos_m): 
        end_time = datetime.now(); duration = (end_time - self.start_time).total_seconds()
        eff = (self.total_sow_dist_m / self.total_dist_m * 100) if self.total_dist_m > 0 else 0
        final_pos_str = f"({final_pos_m[0]:.1f}, {final_pos_m[1]:.1f})m"
        summary = f"\n🏁 MISSION COMPLETE! 🏁\n{'='*25}\n📍 End: {final_pos_str}\n⏰ Time: {duration:.2f}s\n📏 TD: {self.total_dist_m:.1f}m\n🌱 SD: {self.total_sow_dist_m:.1f}m ({eff:.1f}%)\n🚜 VS: {self.sown_v_segs}\n↔️ HS: {self.sown_h_segs}\n{'='*25}"
        print(summary)
        row = [end_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], "FINAL", "End", str(final_pos_m), "", "", "", "", "", "", "", f"{duration:.2f}", f"{self.total_dist_m:.1f}", f"{self.total_sow_dist_m:.1f}", self.sown_v_segs, self.sown_h_segs]
        try:
            with open(self.csv_filename, 'a', newline='', encoding='utf-8') as f: csv.writer(f).writerow(row)
            print(f"💾 Final summary logged to {self.csv_filename}")
        except IOError as e: print(f"❌ CSV Final Err: {e}")


# --- Animation ---
def interpolate_path(path_pts_metric_centers, pts_per_seg=25): 
    if not path_pts_metric_centers or len(path_pts_metric_centers) < 2: return np.array([]), np.array([])
    sx_m, sy_m = [], []
    for i in range(len(path_pts_metric_centers) - 1):
        x0,y0 = path_pts_metric_centers[i]; x1,y1 = path_pts_metric_centers[i+1]
        sx_m.extend(np.linspace(x0, x1, pts_per_seg)); sy_m.extend(np.linspace(y0, y1, pts_per_seg))
    return np.array(sx_m), np.array(sy_m)


def draw_complete_rover_path(path_lanes_list, farm_w_m, farm_b_m, rover_width_m, rover_length_m, ax):
    """
    Draw the complete physical path of the rover regardless of sowing status.
    This is purely geometric - based on farm/rover dimensions only.
    """
    single_brown_color = '#8B4513'
    
    # Convert lane coordinates to metric centers
    path_metric_centers = [
        ((ln_x + 0.5) * rover_width_m, (ln_y + 0.5) * rover_length_m) 
        for ln_x, ln_y in path_lanes_list
    ]
    
    print(f"🔍 Drawing complete rover path: {len(path_metric_centers)-1} segments")
    print(f"🔍 Farm: {farm_w_m}x{farm_b_m}m, Rover: {rover_width_m}x{rover_length_m}m")
    
    # Draw rectangle for EVERY segment (regardless of sowing)
    for i in range(len(path_metric_centers) - 1):
        x1, y1 = path_metric_centers[i]
        x2, y2 = path_metric_centers[i + 1]
        
        # Calculate rectangle dimensions based on movement direction
        if abs(x1 - x2) < 0.001:  # Vertical movement
            rect_x = x1 - rover_width_m / 2
            rect_y = min(y1, y2) - rover_length_m / 2
            rect_width = rover_width_m
            rect_height = abs(y2 - y1) + rover_length_m
        else:  # Horizontal movement
            rect_x = min(x1, x2) - rover_width_m / 2
            rect_y = y1 - rover_length_m / 2
            rect_width = abs(x2 - x1) + rover_width_m
            rect_height = rover_length_m
        
        # Always draw - no conditions based on sowing
        if rect_width > 0 and rect_height > 0:
            path_rect = Rectangle((rect_x, rect_y), rect_width, rect_height, 
                                color=single_brown_color, alpha=0.4, zorder=2)
            ax.add_patch(path_rect)
            print(f"  ✅ Segment {i}: ({x1:.1f},{y1:.1f}) → ({x2:.1f},{y2:.1f}) | Rect: ({rect_x:.1f},{rect_y:.1f}) {rect_width:.1f}x{rect_height:.1f}")
        else:
            print(f"  ❌ Segment {i}: INVALID dimensions - width:{rect_width:.1f}, height:{rect_height:.1f}")
    
    print(f"✅ Complete rover path drawn for ALL {len(path_metric_centers)-1} segments")

def setup_sowing_visualization(path_lanes_list, sow_flags_list, rover_width_m, rover_length_m, ax):
    """
    Setup sowing rectangles that will be animated (only where actually sowing).
    Returns list of rectangles and their progress masks for animation.
    """
    path_metric_centers = [
        ((ln_x + 0.5) * rover_width_m, (ln_y + 0.5) * rover_length_m) 
        for ln_x, ln_y in path_lanes_list
    ]
    
    sown_rectangles = []
    sown_progress_masks = []
    
    print(f"🌱 Setting up sowing visualization for {len(sow_flags_list)} segments")
    
    for i in range(len(path_lanes_list) - 1):
        if i < len(sow_flags_list) and sow_flags_list[i]:
            x1, y1 = path_metric_centers[i]
            x2, y2 = path_metric_centers[i + 1]
            
            # Same rectangle calculation as brown path
            if abs(x1 - x2) < 0.001:  # Vertical movement
                rect_x = x1 - rover_width_m / 2
                rect_y = min(y1, y2) - rover_length_m / 2
                rect_width = rover_width_m
                rect_height = abs(y2 - y1) + rover_length_m
            else:  # Horizontal movement
                rect_x = min(x1, x2) - rover_width_m / 2
                rect_y = y1 - rover_length_m / 2
                rect_width = abs(x2 - x1) + rover_width_m
                rect_height = rover_length_m
            
            if rect_width > 0 and rect_height > 0:
                # Create rectangle that starts with zero dimensions for animation
                if abs(x1 - x2) < 0.001:  # Vertical - start with zero height
                    clip_rect = Rectangle((rect_x, rect_y), rect_width, 0, 
                                        color='#006400', alpha=0.8, zorder=3, visible=False)
                else:  # Horizontal - start with zero width
                    clip_rect = Rectangle((rect_x, rect_y), 0, rect_height, 
                                        color='#006400', alpha=0.8, zorder=3, visible=False)
                
                ax.add_patch(clip_rect)
                sown_rectangles.append((i, clip_rect))
                sown_progress_masks.append({
                    'full_width': rect_width, 
                    'full_height': rect_height, 
                    'is_vertical': abs(x1 - x2) < 0.001, 
                    'base_x': rect_x, 
                    'base_y': rect_y,
                    'start_pos': (x1, y1),
                    'end_pos': (x2, y2)
                })
                print(f"  🌱 Sowing segment {i}: ({x1:.1f},{y1:.1f}) → ({x2:.1f},{y2:.1f})")
            else:
                sown_rectangles.append((i, None))
                sown_progress_masks.append(None)
        else:
            sown_rectangles.append((i, None))
            sown_progress_masks.append(None)
    
    print(f"✅ Sowing visualization setup complete: {sum(1 for _, rect in sown_rectangles if rect is not None)} sowing segments")
    return sown_rectangles, sown_progress_masks

def animate_robot(n_inner_x_sweeps_val, max_lx_idx_val, max_ly_idx_val, title_suffix_str, 
                  path_lanes_list, sow_flags_all_list, 
                  exit_vis_lanes, farm_w_m_val, farm_b_m_val, rover_width_m_val, rover_length_m_val):
    if not path_lanes_list or len(path_lanes_list) < 2: 
        print("❌ Anim Err: Path short."); return None
    if len(sow_flags_all_list) != len(path_lanes_list) -1 : 
        print(f"❌ Anim Err: Mismatch sow_flags ({len(sow_flags_all_list)}) and segments ({len(path_lanes_list)-1}).")
        sow_flags_all_list.extend([False] * (max(0, len(path_lanes_list) - 1 - len(sow_flags_all_list))))

    # Convert lane coordinates to metric centers
    path_metric_centers_list = [((ln_x + 0.5) * rover_width_m_val, (ln_y + 0.5) * rover_length_m_val) for ln_x, ln_y in path_lanes_list]
    exit_vis_metric_center = ((exit_vis_lanes[0] + 0.5) * rover_width_m_val, (exit_vis_lanes[1] + 0.5) * rover_length_m_val)
    
    # Analysis and logging setup
    row_labels_info_list = analyze_optimized_path_sequence(path_lanes_list, n_inner_x_sweeps_val, max_lx_idx_val, max_ly_idx_val, sow_flags_all_list, exit_vis_lanes)    
    seg_labels_info_list = analyze_optimized_path_sequence(path_lanes_list, n_inner_x_sweeps_val, max_lx_idx_val, max_ly_idx_val, sow_flags_all_list, exit_vis_lanes)
    logger_obj = LiveTelemetryLogger(farm_w_m_val, farm_b_m_val, rover_width_m_val, rover_length_m_val, title_suffix_str)
    
    # Smooth path for animation
    smooth_x_m_centers, smooth_y_m_centers = interpolate_path(path_metric_centers_list)
    if smooth_x_m_centers.size == 0: 
        print("❌ Anim Err: Interpolated path empty."); return None
    
    # Movement analysis
    analyses_metric_list = [get_movement_analysis(path_lanes_list, i, max_lx_idx_val, max_ly_idx_val, seg_labels_info_list, sow_flags_all_list, rover_width_m_val, rover_length_m_val) for i in range(len(path_lanes_list) - 1)]
    
    # Setup matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_aspect('equal')
    plot_padding_m = max(rover_width_m_val, rover_length_m_val) * 0.5
    ax.set_xlim(-plot_padding_m, farm_w_m_val + plot_padding_m)
    ax.set_ylim(-plot_padding_m, farm_b_m_val + plot_padding_m)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_title(f'🤖 Farm: {farm_w_m_val}x{farm_b_m_val}m, Rover: {rover_width_m_val}x{rover_length_m_val}m ({title_suffix_str})', fontsize=14, pad=20)
    
    # Draw farm boundary
    ax.add_patch(Rectangle((0,0), farm_w_m_val, farm_b_m_val, fill=False, edgecolor='darkgray', lw=2, zorder=1))
    
    # STEP 1: Draw complete rover path (independent of sowing logic)
    print(f"\n🎨 STEP 1: Drawing complete rover path...")
    draw_complete_rover_path(path_lanes_list, farm_w_m_val, farm_b_m_val, rover_width_m_val, rover_length_m_val, ax)
    
    # STEP 2: Setup sowing visualization (depends on sowing logic)
    print(f"\n🌱 STEP 2: Setting up sowing visualization...")
    sown_rectangles, sown_progress_masks = setup_sowing_visualization(path_lanes_list, sow_flags_all_list, rover_width_m_val, rover_length_m_val, ax)
    
    # Draw row labels (existing code)
    vrow_columns = set()
    first_inner_lx = 1
    last_inner_lx = max_lx_idx_val - 1
    
    for i in range(len(path_lanes_list) - 1):
        lx1, ly1 = path_lanes_list[i]
        lx2, ly2 = path_lanes_list[i + 1]
        is_sown = i < len(sow_flags_all_list) and sow_flags_all_list[i]
        
        if lx1 == lx2 and ly1 != ly2 and is_sown and first_inner_lx <= lx1 <= last_inner_lx:
            vrow_columns.add(lx1)
    
    sorted_vrow_columns = sorted(vrow_columns)
    for i, column_x in enumerate(sorted_vrow_columns):
        tx_m = (column_x + 0.5) * rover_width_m_val
        ty_m = -plot_padding_m * 0.7
        
        label = f'VRow{i+1}'
        ax.text(tx_m, ty_m, label, fontsize=10, color='navy', 
               ha='center', va='center', weight='bold',
               bbox=dict(boxstyle="round,pad=0.3", fc='lightblue', alpha=0.9, ec='navy', lw=2))
    
    # HRow labels (existing code)
    horizontal_y_positions = set()
    for i in range(len(path_lanes_list) - 1):
        x1, y1 = path_lanes_list[i]
        x2, y2 = path_lanes_list[i + 1]
        is_sown = i < len(sow_flags_all_list) and sow_flags_all_list[i]
        
        if y1 == y2 and x1 != x2 and is_sown:
            horizontal_y_positions.add(y1)
    
    sorted_horizontal_positions = sorted(horizontal_y_positions)
    
    if len(sorted_horizontal_positions) >= 1:
        hrow1_y_lane = sorted_horizontal_positions[0]
        hrow1_tx_m = -plot_padding_m * 0.7
        hrow1_ty_m = (hrow1_y_lane + 0.5) * rover_length_m_val
        ax.text(hrow1_tx_m, hrow1_ty_m, 'HRow1', fontsize=10, color='darkgreen', 
               ha='center', va='center', weight='bold', rotation=0,
               bbox=dict(boxstyle="round,pad=0.3", fc='lightgreen', alpha=0.9, ec='darkgreen', lw=2))
    
    if len(sorted_horizontal_positions) >= 2:
        hrow2_y_lane = sorted_horizontal_positions[-1]
        hrow2_tx_m = -plot_padding_m * 0.7
        hrow2_ty_m = (hrow2_y_lane + 0.5) * rover_length_m_val
        ax.text(hrow2_tx_m, hrow2_ty_m, 'HRow2', fontsize=10, color='darkgreen', 
               ha='center', va='center', weight='bold', rotation=0,
               bbox=dict(boxstyle="round,pad=0.3", fc='lightgreen', alpha=0.9, ec='darkgreen', lw=2))
    
    # Robot body and markers (existing code)
    rover_body_width_m = rover_width_m_val 
    rover_body_height_m = rover_length_m_val
    initial_rover_center_m = path_metric_centers_list[0]
    robot_body_patch = Rectangle((initial_rover_center_m[0] - rover_body_width_m/2, initial_rover_center_m[1] - rover_body_height_m/2), 
                                 rover_body_width_m, rover_body_height_m, color='orange', ec='black', lw=1.5, zorder=4)
    ax.add_patch(robot_body_patch)
    
    # Start marker
    start_marker_center_m = path_metric_centers_list[0]
    marker_radius_m = 0.4 * max(rover_width_m_val, rover_length_m_val)
    ax.add_patch(Circle(start_marker_center_m, marker_radius_m, color='blue', fill=True, lw=2.5, zorder=5, alpha=0.5))
    ax.add_patch(Circle(start_marker_center_m, marker_radius_m, color='blue', fill=False, lw=2.5, zorder=5, hatch='//'))
    
    # Exit gate marker
    gate_width = rover_width_m_val * 0.8
    gate_height = rover_length_m_val * 0.3
    
    exit_x, exit_y = exit_vis_lanes[0], exit_vis_lanes[1]
    
    if exit_x == 0:  # Left border
        gate_x = -gate_height/2
        gate_y = exit_vis_metric_center[1] - gate_width/2
        gate_w, gate_h = gate_height, gate_width
    elif exit_x == max_lx_idx_val:  # Right border
        gate_x = farm_w_m_val - gate_height/2
        gate_y = exit_vis_metric_center[1] - gate_width/2
        gate_w, gate_h = gate_height, gate_width
    elif exit_y == 0:  # Bottom border
        gate_x = exit_vis_metric_center[0] - gate_width/2
        gate_y = -gate_height/2
        gate_w, gate_h = gate_width, gate_height
    else:  # Top border
        gate_x = exit_vis_metric_center[0] - gate_width/2
        gate_y = farm_b_m_val - gate_height/2
        gate_w, gate_h = gate_width, gate_height
    
    exit_gate = Rectangle((gate_x, gate_y), gate_w, gate_h, 
                         color='red', alpha=0.8, zorder=5, 
                         edgecolor='darkred', linewidth=2)
    ax.add_patch(exit_gate)
    
    # Legend
    single_brown_color = '#8B4513'
    legend_handles = [
        Line2D([0],[0],c=single_brown_color,lw=10,alpha=0.4,label=f'Complete Rover Path (Rover:{rover_width_m_val}x{rover_length_m_val}m)'), 
        Line2D([0],[0],c='#006400',lw=10,alpha=0.8,label=f'Sown Area (Rover:{rover_width_m_val}x{rover_length_m_val}m)'), 
        Rectangle((0,0), 1, 1, fc='orange', ec='black', label=f'🤖 Rover ({rover_width_m_val}x{rover_length_m_val}m)'),
        Line2D([0],[0],marker='o',mfc='blue',mec='blue',ms=10,ls='None',label=f'🔵 START ({start_marker_center_m[0]:.1f}, {start_marker_center_m[1]:.1f})m'), 
        Rectangle((0,0), 1, 1, fc='red', ec='darkred', label=f'🚪 EXIT GATE ({exit_vis_metric_center[0]:.1f}, {exit_vis_metric_center[1]:.1f})m')
    ]
    ax.legend(handles=legend_handles,loc='upper right',bbox_to_anchor=(1.28,1.02),fontsize=8)
    plt.subplots_adjust(right=0.75)
    
    # Animation setup
    logged_segments_indices = set()
    logger_obj.mission_finalized = False
    total_animation_frames = len(smooth_x_m_centers)
    num_orig_segments = len(path_metric_centers_list) - 1
    pts_per_orig_segment_approx = total_animation_frames // num_orig_segments if num_orig_segments > 0 else total_animation_frames

    def init_animation_func(): 
        robot_body_patch.set_xy((smooth_x_m_centers[0]-rover_body_width_m/2, smooth_y_m_centers[0]-rover_body_height_m/2))
        return [robot_body_patch]
    
    def update_animation_func(frame_idx):
        current_x_center_m = smooth_x_m_centers[frame_idx]
        current_y_center_m = smooth_y_m_centers[frame_idx]
        robot_body_patch.set_xy((current_x_center_m - rover_body_width_m/2, current_y_center_m - rover_body_height_m/2))
        
        current_original_segment_idx = frame_idx // pts_per_orig_segment_approx if pts_per_orig_segment_approx > 0 else num_orig_segments -1
        current_original_segment_idx = min(current_original_segment_idx, num_orig_segments - 1) 

        # Logging
        if current_original_segment_idx != -1 and current_original_segment_idx < len(analyses_metric_list) and current_original_segment_idx not in logged_segments_indices:
            if analyses_metric_list[current_original_segment_idx]:
                 logger_obj.log_movement(current_original_segment_idx + 1, analyses_metric_list[current_original_segment_idx], datetime.now())
            logged_segments_indices.add(current_original_segment_idx)
        
        # Animate sown rectangles gradually
        if current_original_segment_idx >= 0:
            # Make completed segments fully visible
            for i in range(min(current_original_segment_idx, len(sown_rectangles))):
                if i < len(sown_rectangles) and sown_rectangles[i][1] is not None:
                    rect = sown_rectangles[i][1]
                    mask = sown_progress_masks[i]
                    rect.set_visible(True)
                
                    if mask['is_vertical']:
                        rect.set_height(mask['full_height'])
                        rect.set_y(mask['base_y'])
                    else:
                        rect.set_width(mask['full_width'])
                        rect.set_x(mask['base_x'])
        
            # Gradually show current segment
            if (current_original_segment_idx < len(sown_rectangles) and 
                sown_rectangles[current_original_segment_idx][1] is not None and
                current_original_segment_idx < len(sow_flags_all_list) and
                sow_flags_all_list[current_original_segment_idx]):
            
                current_rect = sown_rectangles[current_original_segment_idx][1]
                current_mask = sown_progress_masks[current_original_segment_idx]
                current_rect.set_visible(True)
            
                # Calculate progress within current segment
                segment_start_frame = current_original_segment_idx * pts_per_orig_segment_approx
                frames_in_segment = min(pts_per_orig_segment_approx, total_animation_frames - segment_start_frame)
                progress_in_segment = (frame_idx - segment_start_frame) / frames_in_segment if frames_in_segment > 0 else 1
                progress_in_segment = max(0, min(1, progress_in_segment))
            
                if current_mask['is_vertical']:
                    new_height = current_mask['full_height'] * progress_in_segment
                    current_rect.set_height(new_height)
                
                    start_y, end_y = current_mask['start_pos'][1], current_mask['end_pos'][1]
                    if end_y < start_y:  # Moving from top to bottom
                        new_y = current_mask['base_y'] + current_mask['full_height'] - new_height
                        current_rect.set_y(new_y)
                    else:  # Moving from bottom to top
                        current_rect.set_y(current_mask['base_y'])
                else:
                    new_width = current_mask['full_width'] * progress_in_segment
                    current_rect.set_width(new_width)
                
                    start_x, end_x = current_mask['start_pos'][0], current_mask['end_pos'][0]
                    if end_x < start_x:  # Moving from right to left
                        new_x = current_mask['base_x'] + current_mask['full_width'] - new_width
                        current_rect.set_x(new_x)
                    else:  # Moving from left to right
                        current_rect.set_x(current_mask['base_x'])
        
        # Final frame handling
        if frame_idx >= total_animation_frames - 1 and not logger_obj.mission_finalized: 
            for i, (_, rect) in enumerate(sown_rectangles):
                if rect is not None:
                    rect.set_visible(True)
                    if i < len(sown_progress_masks) and sown_progress_masks[i]:
                        mask = sown_progress_masks[i]
                        if mask['is_vertical']:
                            rect.set_height(mask['full_height'])
                            rect.set_y(mask['base_y'])
                        else:
                            rect.set_width(mask['full_width'])
                            rect.set_x(mask['base_x'])
                        
            for i_log_final_check in range(len(analyses_metric_list)): 
                if i_log_final_check not in logged_segments_indices and analyses_metric_list[i_log_final_check]: 
                    logger_obj.log_movement(i_log_final_check+1, analyses_metric_list[i_log_final_check], datetime.now())
            logger_obj.finalize_mission(path_metric_centers_list[-1]) 
            logger_obj.mission_finalized = True
        
        return [robot_body_patch] + [rect for _, rect in sown_rectangles if rect is not None and rect.get_visible()]

    animation_obj = FuncAnimation(fig, update_animation_func, frames=total_animation_frames, 
                                  init_func=init_animation_func, blit=True, interval=50, repeat=False)
    plt.show()
    return animation_obj




class LiveTelemetryLogger:
    def __init__(self, farm_w_m, farm_b_m, rover_width_m, rover_length_m, exit_info_str):
        self.farm_w_m = farm_w_m; self.farm_b_m = farm_b_m; self.rover_width_m = rover_width_m; self.rover_length_m = rover_length_m
        self.sown_v_segs = 0; self.sown_h_segs = 0
        self.total_dist_m = 0; self.total_sow_dist_m = 0
        self.start_time = datetime.now(); self.csv_filename = "navigation_log.csv"
        if not os.path.exists(self.csv_filename):
            try:
                with open(self.csv_filename, 'w', newline='', encoding='utf-8') as f:
                    w = csv.writer(f); w.writerow(["Timestamp", "Step", "Label", "From (m)", "To (m)", "FromY (m)", "ToY (m)", "Dir", "Action", "FarmType", "Status", "SegDist (m)", "TotalDist (m)", "SownDist (m)", "V_Sown_Segs", "H_Sown_Segs"])
                print(f"💾 CSV created: {self.csv_filename}")
            except IOError as e: print(f"❌ CSV Error: {e}")
        else: print(f"📝 Appending to: {self.csv_filename}")
        hdr = "🤖 FARM ROBOT TELEMETRY 🤖"; print(f"\n{hdr}\n{'='*len(hdr)}\n📊 Farm: {farm_w_m}x{farm_b_m}m, Rover: {rover_width_m}x{rover_length_m}m\n🎯 Exit: {exit_info_str}\n⏰ Start: {self.start_time:%Y-%m-%d %H:%M:%S}\n{'='*len(hdr)}\n🔴 LIVE LOG:\n{'='*len(hdr)}")

    def log_movement(self, step, analysis, time_now): 
        if not analysis: return
        self.total_dist_m += analysis['distance_m']
        if analysis['farming_type'] != 'NONE': 
            self.total_sow_dist_m += analysis['distance_m']
            if 'VERTICAL' in analysis['action'] or '_V' in analysis['farming_type']: self.sown_v_segs += 1
            elif 'HORIZONTAL' in analysis['action'] or '_H' in analysis['farming_type']: self.sown_h_segs += 1
        elapsed = (time_now - self.start_time).total_seconds()
        display_label = analysis['row_sequence_label']
        from_pos_str = f"({analysis['from_pos_m'][0]:.1f}, {analysis['from_pos_m'][1]:.1f})"
        to_pos_str = f"({analysis['to_pos_m'][0]:.1f}, {analysis['to_pos_m'][1]:.1f})"
        print(f"\n⏱️ {time_now:%H:%M:%S.%f}"[:-3] + f" [S{step:02d}] (+{elapsed:.1f}s)\n🏷️ {display_label}\n📍 {from_pos_str} → {to_pos_str} (D:{analysis['distance_m']:.1f}m)\n🧭 Act: {analysis['action']} ({analysis['status']}) | Type: {analysis['farming_type']}\n📊 TD:{self.total_dist_m:.1f}m SD:{self.total_sow_dist_m:.1f}m VS:{self.sown_v_segs} HS:{self.sown_h_segs}\n{'-'*70}")
        row = [time_now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], step, display_label, str(analysis['from_pos_m']), str(analysis['to_pos_m']), f"{analysis['from_row_y_coord_m']:.1f}", f"{analysis['to_row_y_coord_m']:.1f}", analysis['direction'], analysis['action'], analysis['farming_type'], analysis['status'], f"{analysis['distance_m']:.1f}", f"{self.total_dist_m:.1f}", f"{self.total_sow_dist_m:.1f}", self.sown_v_segs, self.sown_h_segs]
        try:
            with open(self.csv_filename, 'a', newline='', encoding='utf-8') as f: csv.writer(f).writerow(row)
        except IOError as e: print(f"❌ CSV Write Err (S{step}): {e}")

    def finalize_mission(self, final_pos_m): 
        end_time = datetime.now(); duration = (end_time - self.start_time).total_seconds()
        eff = (self.total_sow_dist_m / self.total_dist_m * 100) if self.total_dist_m > 0 else 0
        final_pos_str = f"({final_pos_m[0]:.1f}, {final_pos_m[1]:.1f})m"
        summary = f"\n🏁 MISSION COMPLETE! 🏁\n{'='*25}\n📍 End: {final_pos_str}\n⏰ Time: {duration:.2f}s\n📏 TD: {self.total_dist_m:.1f}m\n🌱 SD: {self.total_sow_dist_m:.1f}m ({eff:.1f}%)\n🚜 VS: {self.sown_v_segs}\n↔️ HS: {self.sown_h_segs}\n{'='*25}"
        print(summary)
        row = [end_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], "FINAL", "End", str(final_pos_m), "", "", "", "", "", "", "", f"{duration:.2f}", f"{self.total_dist_m:.1f}", f"{self.total_sow_dist_m:.1f}", self.sown_v_segs, self.sown_h_segs]
        try:
            with open(self.csv_filename, 'a', newline='', encoding='utf-8') as f: csv.writer(f).writerow(row)
            print(f"💾 Final summary logged to {self.csv_filename}")
        except IOError as e: print(f"❌ CSV Final Err: {e}")


def main():
    print("=== 🤖 OPTIMIZED FARM ROBOT TRAVERSAL SYSTEM (v13.0) ===")
    farm_width_m, farm_breadth_m, rover_width_m, rover_length_m = 0.0, 0.0, 0.0, 0.0

    while True: 
        try: farm_width_m = float(input("Enter farm WIDTH (X-axis, e.g., 50 meters): ")); assert farm_width_m > 0; break
        except (ValueError, AssertionError): print("Invalid input. Must be positive number.")
    while True: 
        try: farm_breadth_m = float(input("Enter farm BREADTH (Y-axis, e.g., 50 meters): ")); assert farm_breadth_m > 0; break
        except (ValueError, AssertionError): print("Invalid input. Must be positive number.")
    while True: 
        try: rover_width_m = float(input("Enter rover WIDTH (for lane spacing, e.g., 10 meters): ")); assert rover_width_m > 0; break
        except (ValueError, AssertionError): print("Invalid input. Must be positive number.")
    while True: 
        try: rover_length_m = float(input("Enter rover LENGTH (for coverage depth, e.g., 8 meters): ")); assert rover_length_m > 0; break
        except (ValueError, AssertionError): print("Invalid input. Must be positive number.")

    num_lanes_x = int(farm_width_m / rover_width_m)
    num_lanes_y = int(farm_breadth_m / rover_length_m)

    if num_lanes_x < 3: 
        print(f"❌ Farm width {farm_width_m}m too small for rover width {rover_width_m}m.")
        print(f"   Need at least 3 vertical passes (2 headlands + 1 inner) = {3 * rover_width_m}m minimum farm width.")
        return None
    if num_lanes_y < 1: 
        print(f"❌ Farm breadth {farm_breadth_m}m too small for rover length {rover_length_m}m.")
        print(f"   Need at least 1 horizontal row = {rover_length_m}m minimum farm breadth.")
        return None
        
    max_lane_idx_x = num_lanes_x - 1
    max_lane_idx_y = num_lanes_y - 1
    n_inner_x_sweeps = num_lanes_x - 2 
    if n_inner_x_sweeps < 0: n_inner_x_sweeps = 0 
    
    total_coverage_area = num_lanes_x * num_lanes_y * rover_width_m * rover_length_m
    farm_area = farm_width_m * farm_breadth_m
    coverage_percentage = (total_coverage_area / farm_area * 100) if farm_area > 0 else 0
    
    print(f"\n🚀 OPTIMIZED Grid Analysis for {farm_width_m}x{farm_breadth_m}m farm, Rover: {rover_width_m}x{rover_length_m}m")
    print(f"📊 Grid Layout: {num_lanes_x} vertical passes × {num_lanes_y} horizontal rows")
    print(f"📊 Optimization: Smart start position selection for fastest exit approach")
    print(f"📊 Coverage: {total_coverage_area:.1f}m² of {farm_area:.1f}m² ({coverage_percentage:.1f}%)")
    
    print(f"📊 Productive rows breakdown:")
    print(f"   • Inner vertical sweeps: {n_inner_x_sweeps} (lanes 1 to {max_lane_idx_x-1})" if n_inner_x_sweeps > 0 else "   • No inner vertical sweeps (farm too narrow)")
    print(f"   • Outer vertical headlands: 2 (lanes 0 & {max_lane_idx_x})")
    print(f"   • Horizontal headland rows: {min(2, num_lanes_y)} (covering {min(2, num_lanes_y) * rover_length_m}m breadth)")

    exit_point_lanes, anim_title_suffix_str = None, ""
    print("\nSelect Exit Type:\n1. Corner Exit (select a corner lane)\n2. Custom Boundary Exit (select a boundary lane)")
    nav_choice_str = ""
    while nav_choice_str not in ["1", "2"]: nav_choice_str = input("Choice (1-2): ").strip()
    is_corner_exit_choice = (nav_choice_str == "1")

    if is_corner_exit_choice:
        exit_point_lanes = get_user_choice_corner_lanes(max_lane_idx_x, max_lane_idx_y)
        anim_title_suffix_str = f"Optimized Corner Exit (Lane: {exit_point_lanes})"
    else: 
        exit_point_lanes = get_user_defined_exit_lanes(max_lane_idx_x, max_lane_idx_y)
        anim_title_suffix_str = f"Optimized Custom Exit (Lane: {exit_point_lanes})"
    
    exit_metric_center_display = ((exit_point_lanes[0] + 0.5) * rover_width_m, (exit_point_lanes[1] + 0.5) * rover_length_m)
    
    # Show optimization details
    print(f"🎯 OPTIMIZATION: Starting from VRow1 (lane x=1) for fastest exit approach")

    print(f"ℹ️ Target Exit Lane {exit_point_lanes} (approx. center: {exit_metric_center_display[0]:.1f}m, {exit_metric_center_display[1]:.1f}m)")

    # Use the optimized path generation
    path_data_dict = generate_fixed_path_optimized(n_inner_x_sweeps, max_lane_idx_x, max_lane_idx_y, exit_point_lanes, is_corner_exit_choice)

    path_lanes_list_gen, sow_flags_list_gen = path_data_dict['points_lanes'], path_data_dict['sow_flags']

    if not path_lanes_list_gen or len(path_lanes_list_gen) < 2: 
        print("❌ Optimized path generation failed. Exiting."); return None
    if len(sow_flags_list_gen) != len(path_lanes_list_gen)-1: 
        print(f"❌ CRITICAL MISMATCH: Sow_flags ({len(sow_flags_list_gen)}) vs segments ({len(path_lanes_list_gen)-1}). Exiting.")
        return None

    print(f"\n🎬 OPTIMIZED MISSION SUMMARY:")
    print(f"📊 Farm: {farm_width_m}x{farm_breadth_m}m | Rover: {rover_width_m}x{rover_length_m}m")
    print(f"📊 Grid: {num_lanes_x} vertical passes × {num_lanes_y} horizontal rows")
    print(f"📊 Path: {len(path_lanes_list_gen)} waypoints, {len(sow_flags_list_gen)} segments")
    print(f"📊 Optimization: Smart sequencing for {coverage_percentage:.1f}% coverage with minimal exit time")
    print(f"🎯 OPTIMIZATION: Starting from VRow1 (lane x=1) for fastest exit approach")
    print(f"🎯 OPTIMIZATION: Starting from VRow1 (lane x=1) for fastest exit approach")

    input("\n🎬 Press Enter to start optimized animation & telemetry...")

    final_animation_object = None
    try: 
        final_animation_object = animate_robot(
            n_inner_x_sweeps, max_lane_idx_x, max_lane_idx_y, anim_title_suffix_str, 
            path_lanes_list_gen, sow_flags_list_gen, exit_point_lanes, 
            farm_width_m, farm_breadth_m, rover_width_m, rover_length_m
        )
    except KeyboardInterrupt: print("\n\n⚠️ Animation aborted by user.")
    except Exception as e_anim: 
        print(f"\n❌ An error occurred during animation: {e_anim}")
       
        traceback.print_exc()
    return final_animation_object

# Additional helper function for exit-aware path optimization
def calculate_exit_efficiency_score(current_pos, target_pos, exit_pos, sow_flag):
    """
    Calculate efficiency score for a path segment considering exit proximity and sowing value.
    Lower scores indicate more efficient paths.
    """
    # Distance components
    segment_distance = abs(target_pos[0] - current_pos[0]) + abs(target_pos[1] - current_pos[1])
    exit_distance_before = abs(current_pos[0] - exit_pos[0]) + abs(current_pos[1] - exit_pos[1])
    exit_distance_after = abs(target_pos[0] - exit_pos[0]) + abs(target_pos[1] - exit_pos[1])
    
    # Base score is segment distance
    score = segment_distance
    
    # Penalty for moving away from exit
    if exit_distance_after > exit_distance_before:
        score += (exit_distance_after - exit_distance_before) * 1.5
    
    # Bonus for sowing (productive work)
    if sow_flag:
        score *= 0.8  # 20% bonus for productive segments
    else:
        score *= 1.2  # 20% penalty for non-productive positioning
    
    return score

# Enhanced transition logic for better exit approach
def _add_optimized_headland_transition(current_lane_x, current_lane_y, 
                                     target_lane_x, target_lane_y, 
                                     exit_point_lanes, 
                                     _points_list_lanes, _sow_flags_list, 
                                     segment_label="",
                                     gap_size=1):
    """
    Optimized headland transition that considers exit proximity for better efficiency.
    """
    ex_lane_x, ey_lane_y = exit_point_lanes
    
    # Check if we can take a more direct route to the exit
    direct_exit_distance = abs(current_lane_x - ex_lane_x) + abs(current_lane_y - ey_lane_y)
    via_target_exit_distance = abs(current_lane_x - target_lane_x) + abs(target_lane_x - ex_lane_x) + abs(current_lane_y - target_lane_y) + abs(target_lane_y - ey_lane_y)
    
    # If direct route is significantly shorter and we're close to exit, consider optimization
    if direct_exit_distance < via_target_exit_distance * 0.7 and direct_exit_distance <= 3:
        # Direct approach optimization
        if current_lane_x != ex_lane_x:
            # Move toward exit X first
            intermediate_x = ex_lane_x + (1 if ex_lane_x > 0 else -1) if abs(current_lane_x - ex_lane_x) > 1 else ex_lane_x
            _commit_point_to_path(_points_list_lanes, _sow_flags_list, (intermediate_x, current_lane_y), True, f"{segment_label}_DirectApproachX")
            current_lane_x = intermediate_x
        
        if current_lane_y != ey_lane_y:
            # Move toward exit Y
            _commit_point_to_path(_points_list_lanes, _sow_flags_list, (current_lane_x, ey_lane_y), True, f"{segment_label}_DirectApproachY")
            current_lane_y = ey_lane_y
        
        return current_lane_x, current_lane_y
    else:
        # Standard transition with exit awareness
        return _add_headland_segment_custom_exit_with_gaps(
            current_lane_x, current_lane_y, target_lane_x, target_lane_y,
            exit_point_lanes, _points_list_lanes, _sow_flags_list,
            segment_label, gap_size=gap_size
        )


if __name__ == "__main__":
    run_animation_main_obj = main()
    if run_animation_main_obj: 
        print("\n✅ Animation process completed or started successfully.")
        print("📁 Check 'navigation_log.csv' for telemetry data.")
    else: 
        print("\n🔴 Animation did not complete or was not started due to an error.")






