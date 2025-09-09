"""
Camera projection utilities for converting 2D detections to 3D positions
"""
import numpy as np
from geometry_msgs.msg import Point, PointStamped
from sensor_msgs.msg import CameraInfo
from vision_msgs.msg import Detection2D
import tf2_geometry_msgs
import math


class CameraProjection:
    """Handle camera projection and ground plane intersection"""
    
    def __init__(self):
        self.camera_info = None
        self.camera_matrix = None
        self.ground_plane_height = 0.0  # Assume ground at z=0
        self.camera_height = 1.0  # Default camera height above ground
        self.camera_tilt_angle = 0.0  # Camera tilt in radians (0 = looking straight ahead)
        
    def set_camera_info(self, camera_info: CameraInfo):
        """Set camera calibration parameters"""
        self.camera_info = camera_info
        self.camera_matrix = np.array([
            [camera_info.k[0], camera_info.k[1], camera_info.k[2]],
            [camera_info.k[3], camera_info.k[4], camera_info.k[5]],
            [camera_info.k[6], camera_info.k[7], camera_info.k[8]]
        ])
        
    def set_ground_plane_params(self, camera_height: float, tilt_angle: float = 0.0):
        """Set ground plane intersection parameters"""
        self.camera_height = camera_height
        self.camera_tilt_angle = tilt_angle
        
    def pixel_to_3d_with_depth(self, u: float, v: float, depth: float) -> Point:
        """Convert pixel coordinates to 3D point using depth information"""
        if self.camera_matrix is None:
            raise ValueError("Camera info not set")
            
        # Intrinsic parameters
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        # Back-project to 3D
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        
        point = Point()
        point.x = x
        point.y = y
        point.z = z
        return point
        
    def pixel_to_ground_plane(self, u: float, v: float) -> Point:
        """
        Convert pixel coordinates to ground plane intersection
        Assumes camera is looking down at an angle towards the ground
        """
        if self.camera_matrix is None:
            raise ValueError("Camera info not set")
            
        # Intrinsic parameters
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        # Create ray direction in camera frame
        ray_dir = np.array([
            (u - cx) / fx,
            (v - cy) / fy,
            1.0
        ])
        ray_dir = ray_dir / np.linalg.norm(ray_dir)
        
        # Apply camera tilt
        cos_tilt = math.cos(self.camera_tilt_angle)
        sin_tilt = math.sin(self.camera_tilt_angle)
        
        # Rotation matrix for tilt around x-axis
        tilt_matrix = np.array([
            [1.0, 0.0, 0.0],
            [0.0, cos_tilt, -sin_tilt],
            [0.0, sin_tilt, cos_tilt]
        ])
        
        ray_dir = tilt_matrix @ ray_dir
        
        # Camera position (assuming camera is at height above ground, looking forward)
        camera_pos = np.array([0.0, 0.0, self.camera_height])
        
        # Ground plane intersection
        # Ray: P = camera_pos + t * ray_dir
        # Plane: z = ground_plane_height
        if abs(ray_dir[2]) < 1e-6:
            # Ray is parallel to ground plane
            raise ValueError("Ray parallel to ground plane")
            
        t = (self.ground_plane_height - camera_pos[2]) / ray_dir[2]
        
        if t < 0:
            # Intersection behind camera
            raise ValueError("Ground intersection behind camera")
            
        intersection = camera_pos + t * ray_dir
        
        point = Point()
        point.x = intersection[0]
        point.y = intersection[1]
        point.z = intersection[2]
        return point
        
    def get_detection_bottom_center(self, detection: Detection2D) -> tuple:
        """Get bottom center pixel coordinates of detection bounding box"""
        bbox = detection.bbox
        center_u = bbox.center.x
        bottom_v = bbox.center.y + bbox.size_y / 2.0
        return center_u, bottom_v
        
    def detection_to_3d_point(self, detection: Detection2D, depth_value: float = None) -> Point:
        """
        Convert detection to 3D point
        If depth_value is provided, use depth projection
        Otherwise, use ground plane intersection
        """
        center_u, bottom_v = self.get_detection_bottom_center(detection)
        
        if depth_value is not None and depth_value > 0:
            return self.pixel_to_3d_with_depth(center_u, bottom_v, depth_value)
        else:
            return self.pixel_to_ground_plane(center_u, bottom_v)


def extract_depth_at_pixel(depth_image, u: int, v: int, search_radius: int = 5) -> float:
    """
    Extract depth value at pixel with fallback to nearby pixels if invalid
    Returns depth in meters, or None if no valid depth found
    """
    if depth_image is None:
        return None
        
    height, width = depth_image.shape[:2]
    
    # Clamp coordinates
    u = max(0, min(width - 1, int(u)))
    v = max(0, min(height - 1, int(v)))
    
    # Try center pixel first
    depth = depth_image[v, u]
    if depth > 0:
        return depth / 1000.0  # Convert mm to meters (assuming uint16 depth in mm)
        
    # Search in expanding squares
    for radius in range(1, search_radius + 1):
        for dv in range(-radius, radius + 1):
            for du in range(-radius, radius + 1):
                if abs(dv) != radius and abs(du) != radius:
                    continue  # Only check perimeter
                    
                check_u = u + du
                check_v = v + dv
                
                if 0 <= check_u < width and 0 <= check_v < height:
                    depth = depth_image[check_v, check_u]
                    if depth > 0:
                        return depth / 1000.0
                        
    return None  # No valid depth found


def validate_3d_point(point: Point, max_distance: float = 50.0) -> bool:
    """Validate if 3D point is reasonable for robot navigation"""
    # Check for NaN or inf
    if not (math.isfinite(point.x) and math.isfinite(point.y) and math.isfinite(point.z)):
        return False
        
    # Check distance from origin
    distance = math.sqrt(point.x**2 + point.y**2 + point.z**2)
    if distance > max_distance:
        return False
        
    # Check if point is above ground (reasonable height)
    if point.z < -1.0 or point.z > 3.0:
        return False
        
    return True
