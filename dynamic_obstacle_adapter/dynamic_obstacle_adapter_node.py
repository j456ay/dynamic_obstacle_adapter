#!/usr/bin/env python3
"""
Dynamic Obstacle Adapter Node for ROS2 Nav2 Integration
Converts YOLO detections to dynamic obstacles for navigation
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import numpy as np
import cv2
from cv_bridge import CvBridge
import tf2_ros
import tf2_geometry_msgs
from threading import Lock
import time
import math

# ROS2 Messages
from std_msgs.msg import Header, Empty
from geometry_msgs.msg import Point, PointStamped, TransformStamped
from sensor_msgs.msg import PointCloud2, PointField, CameraInfo, Image
from vision_msgs.msg import Detection2DArray, Detection2D
from nav_msgs.msg import Path
from visualization_msgs.msg import MarkerArray, Marker

# Custom utilities
from .geometry_utils import (
    check_path_obstacle_intersection,
    apply_exponential_smoothing,
    create_circular_footprint_points,
    calculate_obstacle_score
)
from .projection_utils import CameraProjection, extract_depth_at_pixel, validate_3d_point


class DynamicObstacle:
    """Represents a tracked dynamic obstacle"""
    
    def __init__(self, class_id: str, position: Point, timestamp: float):
        self.class_id = class_id
        self.position = position
        self.smoothed_position = Point()
        self.smoothed_position.x = position.x
        self.smoothed_position.y = position.y
        self.smoothed_position.z = position.z
        self.last_update = timestamp
        self.detection_count = 1
        self.velocity = Point()
        self.radius = 0.5
        self.weight = 1.0
        self.threat_score = 0.0


class DynamicObstacleAdapterNode(Node):
    """Main node for dynamic obstacle adaptation"""
    
    def __init__(self):
        super().__init__('dynamic_obstacle_adapter_node')
        
        # Thread safety
        self.lock = Lock()
        
        # CV Bridge for image processing
        self.cv_bridge = CvBridge()
        
        # TF2 components
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Camera projection handler
        self.camera_projection = CameraProjection()
        
        # Obstacle tracking
        self.obstacles = {}  # Dict[str, DynamicObstacle]
        self.obstacle_id_counter = 0
        
        # Current path for collision checking
        self.current_path = None
        self.path_timestamp = 0.0
        
        # Depth image for 3D projection
        self.current_depth_image = None
        self.depth_timestamp = 0.0
        
        # Initialize parameters
        self._declare_parameters()
        self._load_parameters()
        
        # Setup QoS profiles
        self.sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )
        
        self.reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # Publishers
        self.obstacle_cloud_pub = self.create_publisher(
            PointCloud2, 
            '/dynamic_obstacles', 
            self.reliable_qos
        )
        
        self.replan_pub = self.create_publisher(
            Empty,
            '/replan',
            self.reliable_qos
        )
        
        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/dynamic_obstacle_markers',
            self.reliable_qos
        )
        
        # Subscribers
        self.detection_sub = self.create_subscription(
            Detection2DArray,
            self.detection_topic,
            self.detection_callback,
            self.reliable_qos
        )
        
        if self.camera_info_topic:
            self.camera_info_sub = self.create_subscription(
                CameraInfo,
                self.camera_info_topic,
                self.camera_info_callback,
                self.reliable_qos
            )
            
        if self.use_depth and self.depth_image_topic:
            self.depth_sub = self.create_subscription(
                Image,
                self.depth_image_topic,
                self.depth_callback,
                self.sensor_qos
            )
        
        self.path_sub = self.create_subscription(
            Path,
            '/selected_path',
            self.path_callback,
            self.reliable_qos
        )
        
        # Timer for publishing obstacles
        self.publish_timer = self.create_timer(
            1.0 / self.publish_rate_hz,
            self.publish_obstacles
        )
        
        # Timer for obstacle cleanup
        self.cleanup_timer = self.create_timer(2.0, self.cleanup_old_obstacles)
        
        self.get_logger().info(f"Dynamic Obstacle Adapter initialized")
        self.get_logger().info(f"Use depth: {self.use_depth}")
        self.get_logger().info(f"Target frame: {self.target_frame}")
        self.get_logger().info(f"Class parameters: {self.class_params}")
        
    def _declare_parameters(self):
        """Declare ROS parameters"""
        # Core parameters
        self.declare_parameter('use_depth', False)
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('depth_image_topic', '/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('detection_topic', '/detections')
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('target_frame', 'map')
        self.declare_parameter('publish_rate_hz', 10.0)
        
        # Ground plane parameters (used when depth is not available)
        self.declare_parameter('camera_height', 1.0)
        self.declare_parameter('camera_tilt_angle', 0.1)  # radians
        
        # Smoothing and tracking parameters
        self.declare_parameter('smoothing_alpha', 0.7)
        self.declare_parameter('obstacle_timeout', 2.0)
        self.declare_parameter('max_detection_distance', 10.0)
        
        # Replanning parameters
        self.declare_parameter('replan_distance_threshold', 1.2)
        self.declare_parameter('replan_score_threshold', 10.0)
        self.declare_parameter('path_lookahead_distance', 5.0)
        
        # Class-specific parameters
        self.declare_parameter('class_params.person.radius', 0.8)
        self.declare_parameter('class_params.person.weight', 1.0)
        self.declare_parameter('class_params.bicycle.radius', 1.0)
        self.declare_parameter('class_params.bicycle.weight', 0.8)
        self.declare_parameter('class_params.car.radius', 2.0)
        self.declare_parameter('class_params.car.weight', 1.2)
        self.declare_parameter('class_params.motorcycle.radius', 1.2)
        self.declare_parameter('class_params.motorcycle.weight', 0.9)
        self.declare_parameter('class_params.default.radius', 0.5)
        self.declare_parameter('class_params.default.weight', 1.0)
        
    def _load_parameters(self):
        """Load parameters from ROS parameter server"""
        self.use_depth = self.get_parameter('use_depth').get_parameter_value().bool_value
        self.camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        self.depth_image_topic = self.get_parameter('depth_image_topic').get_parameter_value().string_value
        self.detection_topic = self.get_parameter('detection_topic').get_parameter_value().string_value
        self.camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value
        self.target_frame = self.get_parameter('target_frame').get_parameter_value().string_value
        self.publish_rate_hz = self.get_parameter('publish_rate_hz').get_parameter_value().double_value
        
        # Ground plane parameters
        self.camera_height = self.get_parameter('camera_height').get_parameter_value().double_value
        self.camera_tilt_angle = self.get_parameter('camera_tilt_angle').get_parameter_value().double_value
        
        # Tracking parameters
        self.smoothing_alpha = self.get_parameter('smoothing_alpha').get_parameter_value().double_value
        self.obstacle_timeout = self.get_parameter('obstacle_timeout').get_parameter_value().double_value
        self.max_detection_distance = self.get_parameter('max_detection_distance').get_parameter_value().double_value
        
        # Replanning parameters
        self.replan_distance_threshold = self.get_parameter('replan_distance_threshold').get_parameter_value().double_value
        self.replan_score_threshold = self.get_parameter('replan_score_threshold').get_parameter_value().double_value
        self.path_lookahead_distance = self.get_parameter('path_lookahead_distance').get_parameter_value().double_value
        
        # Load class parameters
        self.class_params = {}
        class_names = ['person', 'bicycle', 'car', 'motorcycle', 'default']
        
        for class_name in class_names:
            try:
                radius = self.get_parameter(f'class_params.{class_name}.radius').get_parameter_value().double_value
                weight = self.get_parameter(f'class_params.{class_name}.weight').get_parameter_value().double_value
                self.class_params[class_name] = {'radius': radius, 'weight': weight}
            except Exception as e:
                self.get_logger().warn(f"Failed to load parameters for {class_name}: {e}")
                self.class_params[class_name] = {'radius': 0.5, 'weight': 1.0}
        
        # Set up camera projection
        self.camera_projection.set_ground_plane_params(self.camera_height, self.camera_tilt_angle)
        
    def camera_info_callback(self, msg: CameraInfo):
        """Handle camera info updates"""
        try:
            self.camera_projection.set_camera_info(msg)
            self.get_logger().debug("Camera info updated")
        except Exception as e:
            self.get_logger().error(f"Failed to process camera info: {e}")
            
    def depth_callback(self, msg: Image):
        """Handle depth image updates"""
        try:
            if msg.encoding == '16UC1':
                self.current_depth_image = self.cv_bridge.imgmsg_to_cv2(msg, '16UC1')
            elif msg.encoding == '32FC1':
                self.current_depth_image = self.cv_bridge.imgmsg_to_cv2(msg, '32FC1')
            else:
                self.get_logger().warn(f"Unsupported depth encoding: {msg.encoding}")
                return
                
            self.depth_timestamp = time.time()
            self.get_logger().debug("Depth image updated")
            
        except Exception as e:
            self.get_logger().error(f"Failed to process depth image: {e}")
            
    def path_callback(self, msg: Path):
        """Handle current path updates"""
        with self.lock:
            self.current_path = msg
            self.path_timestamp = time.time()
            
    def detection_callback(self, msg: Detection2DArray):
        """Main detection processing callback"""
        try:
            current_time = time.time()
            
            # Process each detection
            for detection in msg.detections:
                self._process_single_detection(detection, current_time)
                
        except Exception as e:
            self.get_logger().error(f"Failed to process detections: {e}")
            
    def _process_single_detection(self, detection: Detection2D, timestamp: float):
        """Process a single detection and update obstacle tracking"""
        try:
            # Get class name (use first hypothesis if available)
            class_name = 'default'
            if detection.results and len(detection.results) > 0:
                class_name = detection.results[0].hypothesis.class_id
                
            # Get class parameters
            class_params = self.class_params.get(class_name, self.class_params['default'])
            
            # Convert detection to 3D point
            point_3d = self._detection_to_3d_point(detection)
            if point_3d is None:
                return
                
            # Transform to target frame
            point_map = self._transform_point_to_target_frame(point_3d, timestamp)
            if point_map is None:
                return
                
            # Validate point
            if not validate_3d_point(point_map, self.max_detection_distance):
                return
                
            # Update or create obstacle
            obstacle_id = self._find_or_create_obstacle(point_map, class_name, timestamp)
            if obstacle_id is not None:
                obstacle = self.obstacles[obstacle_id]
                self._update_obstacle(obstacle, point_map, class_params, timestamp)
                
        except Exception as e:
            self.get_logger().error(f"Failed to process detection: {e}")
            
    def _detection_to_3d_point(self, detection: Detection2D) -> Point:
        """Convert 2D detection to 3D point"""
        try:
            center_u, bottom_v = self.camera_projection.get_detection_bottom_center(detection)
            
            # Try depth-based projection first
            if self.use_depth and self.current_depth_image is not None:
                depth_value = extract_depth_at_pixel(
                    self.current_depth_image, 
                    int(center_u), 
                    int(bottom_v)
                )
                
                if depth_value is not None:
                    return self.camera_projection.pixel_to_3d_with_depth(
                        center_u, bottom_v, depth_value
                    )
            
            # Fallback to ground plane projection
            return self.camera_projection.pixel_to_ground_plane(center_u, bottom_v)
            
        except Exception as e:
            self.get_logger().warn(f"Failed to convert detection to 3D: {e}")
            return None
            
    def _transform_point_to_target_frame(self, point: Point, timestamp: float) -> Point:
        """Transform point from camera frame to target frame"""
        try:
            point_stamped = PointStamped()
            point_stamped.header.frame_id = self.camera_frame
            point_stamped.header.stamp = self.get_clock().now().to_msg()
            point_stamped.point = point
            
            # Get transform
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.camera_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            
            # Transform point
            transformed_point = tf2_geometry_msgs.do_transform_point(point_stamped, transform)
            return transformed_point.point
            
        except Exception as e:
            self.get_logger().warn(f"Failed to transform point: {e}")
            return None
            
    def _find_or_create_obstacle(self, position: Point, class_name: str, timestamp: float) -> str:
        """Find existing obstacle or create new one"""
        with self.lock:
            # Look for existing obstacle within reasonable distance
            min_distance = float('inf')
            closest_id = None
            
            for obs_id, obstacle in self.obstacles.items():
                if obstacle.class_id != class_name:
                    continue
                    
                distance = math.sqrt(
                    (position.x - obstacle.smoothed_position.x)**2 + 
                    (position.y - obstacle.smoothed_position.y)**2
                )
                
                if distance < min_distance and distance < 1.0:  # 1m association threshold
                    min_distance = distance
                    closest_id = obs_id
                    
            if closest_id is not None:
                return closest_id
                
            # Create new obstacle
            new_id = f"obstacle_{self.obstacle_id_counter}"
            self.obstacle_id_counter += 1
            
            self.obstacles[new_id] = DynamicObstacle(class_name, position, timestamp)
            return new_id
            
    def _update_obstacle(self, obstacle: DynamicObstacle, position: Point, 
                        class_params: dict, timestamp: float):
        """Update obstacle with new detection"""
        # Calculate velocity
        dt = timestamp - obstacle.last_update
        if dt > 0:
            obstacle.velocity.x = (position.x - obstacle.smoothed_position.x) / dt
            obstacle.velocity.y = (position.y - obstacle.smoothed_position.y) / dt
            
        # Apply smoothing
        obstacle.smoothed_position.x = apply_exponential_smoothing(
            obstacle.smoothed_position.x, position.x, self.smoothing_alpha
        )
        obstacle.smoothed_position.y = apply_exponential_smoothing(
            obstacle.smoothed_position.y, position.y, self.smoothing_alpha
        )
        obstacle.smoothed_position.z = apply_exponential_smoothing(
            obstacle.smoothed_position.z, position.z, self.smoothing_alpha
        )
        
        # Update parameters
        obstacle.radius = class_params['radius']
        obstacle.weight = class_params['weight']
        obstacle.last_update = timestamp
        obstacle.detection_count += 1
        
        # Calculate threat score
        if self.current_path is not None:
            intersects, min_distance, _ = check_path_obstacle_intersection(
                self.current_path,
                obstacle.smoothed_position,
                obstacle.radius,
                self.path_lookahead_distance
            )
            
            obstacle.threat_score = calculate_obstacle_score(
                min_distance, obstacle.radius, obstacle.weight
            )
        else:
            obstacle.threat_score = 0.0
            
    def cleanup_old_obstacles(self):
        """Remove obstacles that haven't been updated recently"""
        current_time = time.time()
        
        with self.lock:
            obstacles_to_remove = []
            
            for obs_id, obstacle in self.obstacles.items():
                if current_time - obstacle.last_update > self.obstacle_timeout:
                    obstacles_to_remove.append(obs_id)
                    
            for obs_id in obstacles_to_remove:
                del self.obstacles[obs_id]
                
            if obstacles_to_remove:
                self.get_logger().debug(f"Removed {len(obstacles_to_remove)} old obstacles")
                
    def publish_obstacles(self):
        """Publish dynamic obstacles as point cloud and markers"""
        try:
            current_time = time.time()
            
            with self.lock:
                obstacles_copy = dict(self.obstacles)
                
            if not obstacles_copy:
                return
                
            # Create point cloud
            self._publish_obstacle_cloud(obstacles_copy, current_time)
            
            # Create visualization markers
            self._publish_obstacle_markers(obstacles_copy, current_time)
            
            # Check if replanning is needed
            self._check_replanning_conditions(obstacles_copy)
            
        except Exception as e:
            self.get_logger().error(f"Failed to publish obstacles: {e}")
            
    def _publish_obstacle_cloud(self, obstacles: dict, timestamp: float):
        """Publish obstacles as PointCloud2"""
        points = []
        
        for obstacle in obstacles.values():
            # Add center point
            points.append([
                obstacle.smoothed_position.x,
                obstacle.smoothed_position.y,
                obstacle.smoothed_position.z,
                obstacle.weight,
                obstacle.threat_score
            ])
            
            # Add footprint points for better coverage
            footprint_points = create_circular_footprint_points(
                obstacle.smoothed_position, obstacle.radius, 8
            )
            
            for fp_point in footprint_points:
                points.append([
                    fp_point.x,
                    fp_point.y,
                    fp_point.z,
                    obstacle.weight,
                    obstacle.threat_score
                ])
                
        if not points:
            return
            
        # Create PointCloud2 message
        cloud_msg = PointCloud2()
        cloud_msg.header.frame_id = self.target_frame
        cloud_msg.header.stamp = self.get_clock().now().to_msg()
        
        cloud_msg.height = 1
        cloud_msg.width = len(points)
        cloud_msg.is_dense = True
        cloud_msg.is_bigendian = False
        
        # Define fields
        cloud_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='weight', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='score', offset=16, datatype=PointField.FLOAT32, count=1),
        ]
        
        cloud_msg.point_step = 20
        cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
        
        # Pack data
        import struct
        cloud_data = []
        for point in points:
            cloud_data.extend(struct.pack('fffff', *point))
            
        cloud_msg.data = bytes(cloud_data)
        
        self.obstacle_cloud_pub.publish(cloud_msg)
        
    def _publish_obstacle_markers(self, obstacles: dict, timestamp: float):
        """Publish visualization markers for obstacles"""
        marker_array = MarkerArray()
        marker_id = 0
        
        for obs_id, obstacle in obstacles.items():
            # Obstacle cylinder marker
            marker = Marker()
            marker.header.frame_id = self.target_frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "dynamic_obstacles"
            marker.id = marker_id
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            
            marker.pose.position = obstacle.smoothed_position
            marker.pose.orientation.w = 1.0
            
            marker.scale.x = obstacle.radius * 2.0
            marker.scale.y = obstacle.radius * 2.0
            marker.scale.z = 0.5
            
            # Color based on threat score
            if obstacle.threat_score > self.replan_score_threshold:
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
            elif obstacle.threat_score > self.replan_score_threshold * 0.5:
                marker.color.r = 1.0
                marker.color.g = 0.5
                marker.color.b = 0.0
            else:
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                
            marker.color.a = 0.7
            marker.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()
            
            marker_array.markers.append(marker)
            marker_id += 1
            
            # Text marker for class and score
            text_marker = Marker()
            text_marker.header = marker.header
            text_marker.ns = "obstacle_labels"
            text_marker.id = marker_id
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            
            text_marker.pose.position.x = obstacle.smoothed_position.x
            text_marker.pose.position.y = obstacle.smoothed_position.y
            text_marker.pose.position.z = obstacle.smoothed_position.z + 0.5
            text_marker.pose.orientation.w = 1.0
            
            text_marker.scale.z = 0.3
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            
            text_marker.text = f"{obstacle.class_id}\nScore: {obstacle.threat_score:.1f}"
            text_marker.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()
            
            marker_array.markers.append(text_marker)
            marker_id += 1
            
        self.marker_pub.publish(marker_array)
        
    def _check_replanning_conditions(self, obstacles: dict):
        """Check if replanning should be triggered"""
        if self.current_path is None:
            return
            
        should_replan = False
        max_threat_score = 0.0
        
        for obstacle in obstacles.values():
            if obstacle.threat_score > max_threat_score:
                max_threat_score = obstacle.threat_score
                
            # Check distance-based replanning
            intersects, min_distance, _ = check_path_obstacle_intersection(
                self.current_path,
                obstacle.smoothed_position,
                obstacle.radius,
                self.path_lookahead_distance
            )
            
            if intersects or min_distance < self.replan_distance_threshold:
                should_replan = True
                break
                
        # Check score-based replanning
        if max_threat_score > self.replan_score_threshold:
            should_replan = True
            
        if should_replan:
            self.get_logger().info(f"Triggering replan - max threat: {max_threat_score:.1f}")
            replan_msg = Empty()
            self.replan_pub.publish(replan_msg)


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = DynamicObstacleAdapterNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
