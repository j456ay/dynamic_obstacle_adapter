"""
Geometry utilities for dynamic obstacle processing
"""
import numpy as np
from geometry_msgs.msg import Point
from nav_msgs.msg import Path
import math


def point_distance(p1: Point, p2: Point) -> float:
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)


def point_to_line_distance(point: Point, line_start: Point, line_end: Point) -> float:
    """Calculate perpendicular distance from point to line segment"""
    # Vector from line_start to line_end
    line_vec = np.array([line_end.x - line_start.x, line_end.y - line_start.y])
    # Vector from line_start to point
    point_vec = np.array([point.x - line_start.x, point.y - line_start.y])
    
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return np.linalg.norm(point_vec)
    
    # Project point onto line
    line_unit = line_vec / line_len
    proj_length = np.dot(point_vec, line_unit)
    
    # Clamp to line segment
    proj_length = max(0, min(line_len, proj_length))
    
    # Find closest point on line segment
    closest_point = line_start
    closest_point.x += proj_length * line_unit[0]
    closest_point.y += proj_length * line_unit[1]
    
    return point_distance(point, closest_point)


def check_path_obstacle_intersection(path: Path, obstacle_pos: Point, 
                                   obstacle_radius: float, 
                                   lookahead_distance: float = 5.0) -> tuple:
    """
    Check if obstacle intersects with robot path within lookahead distance
    Returns: (intersects: bool, min_distance: float, intersection_point_idx: int)
    """
    if not path.poses or len(path.poses) < 2:
        return False, float('inf'), -1
    
    min_distance = float('inf')
    intersects = False
    intersection_idx = -1
    
    # Check path segments within lookahead distance
    for i in range(len(path.poses) - 1):
        start_pose = path.poses[i]
        end_pose = path.poses[i + 1]
        
        # Check if this segment is within lookahead
        start_point = Point()
        start_point.x = start_pose.pose.position.x
        start_point.y = start_pose.pose.position.y
        
        if i == 0:  # First segment, check from robot position
            segment_distance = 0.0
        else:
            prev_point = Point()
            prev_point.x = path.poses[0].pose.position.x
            prev_point.y = path.poses[0].pose.position.y
            segment_distance = point_distance(prev_point, start_point)
        
        if segment_distance > lookahead_distance:
            break
        
        # Calculate distance from obstacle to this path segment
        end_point = Point()
        end_point.x = end_pose.pose.position.x
        end_point.y = end_pose.pose.position.y
        
        distance = point_to_line_distance(obstacle_pos, start_point, end_point)
        
        if distance < min_distance:
            min_distance = distance
            
        # Check if obstacle intersects with path (considering robot footprint)
        if distance <= obstacle_radius:
            intersects = True
            intersection_idx = i
    
    return intersects, min_distance, intersection_idx


def apply_exponential_smoothing(current_value: float, new_value: float, alpha: float) -> float:
    """Apply exponential moving average smoothing"""
    return alpha * new_value + (1.0 - alpha) * current_value


def create_circular_footprint_points(center: Point, radius: float, num_points: int = 16) -> list:
    """Create points around a circular footprint for visualization"""
    points = []
    for i in range(num_points):
        angle = 2.0 * math.pi * i / num_points
        point = Point()
        point.x = center.x + radius * math.cos(angle)
        point.y = center.y + radius * math.sin(angle)
        point.z = 0.0
        points.append(point)
    return points


def calculate_obstacle_score(distance: float, radius: float, weight: float) -> float:
    """
    Calculate obstacle threat score based on distance, size, and class weight
    Higher score = more dangerous
    """
    if distance <= radius:
        return weight * 100.0  # Maximum threat if overlapping
    
    # Inverse quadratic falloff
    effective_distance = distance - radius
    if effective_distance <= 0.1:
        effective_distance = 0.1
    
    score = weight * (1.0 / (effective_distance ** 2))
    return min(score, weight * 100.0)  # Cap at maximum threat
