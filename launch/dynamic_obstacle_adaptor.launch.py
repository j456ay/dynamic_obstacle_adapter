#!/usr/bin/env python3
"""
Launch file for dynamic obstacle adapter
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    
    # Package directory
    pkg_dir = FindPackageShare('dynamic_obstacle_adapter')
    
    # Launch arguments
    use_depth_arg = DeclareLaunchArgument(
        'use_depth',
        default_value='true',
        description='Whether to use depth camera for 3D projection'
    )
    
    camera_info_topic_arg = DeclareLaunchArgument(
        'camera_info_topic',
        default_value='/camera/color/camera_info',
        description='Camera info topic'
    )
    
    depth_image_topic_arg = DeclareLaunchArgument(
        'depth_image_topic',
        default_value='/camera/aligned_depth_to_color/image_raw',
        description='Depth image topic'
    )
    
    detection_topic_arg = DeclareLaunchArgument(
        'detection_topic',
        default_value='/detections',
        description='YOLO detection topic'
    )
    
    camera_frame_arg = DeclareLaunchArgument(
        'camera_frame',
        default_value='camera_color_optical_frame',
        description='Camera optical frame ID'
    )
    
    target_frame_arg = DeclareLaunchArgument(
        'target_frame',
        default_value='map',
        description='Target frame for obstacle coordinates'
    )
    
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution([
            pkg_dir, 'config', 'dynamic_obstacle_params.yaml'
        ]),
        description='Configuration file path'
    )
    
    # Node definition
    dynamic_obstacle_node = Node(
        package='dynamic_obstacle_adapter',
        executable='dynamic_obstacle_adapter_node',
        name='dynamic_obstacle_adapter_node',
        parameters=[
            LaunchConfiguration('config_file'),
            {
                'use_depth': LaunchConfiguration('use_depth'),
                'camera_info_topic': LaunchConfiguration('camera_info_topic'),
                'depth_image_topic': LaunchConfiguration('depth_image_topic'),
                'detection_topic': LaunchConfiguration('detection_topic'),
                'camera_frame': LaunchConfiguration('camera_frame'),
                'target_frame': LaunchConfiguration('target_frame'),
            }
        ],
        output='screen',
        emulate_tty=True,
        respawn=True,
        respawn_delay=2.0
    )
    
    return LaunchDescription([
        use_depth_arg,
        camera_info_topic_arg,
        depth_image_topic_arg,
        detection_topic_arg,
        camera_frame_arg,
        target_frame_arg,
        config_file_arg,
        dynamic_obstacle_node
    ])
