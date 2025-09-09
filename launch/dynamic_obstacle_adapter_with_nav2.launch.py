#!/usr/bin/env python3
"""
Launch file for dynamic obstacle adapter with Nav2 integration
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node, SetRemap
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    
    # Package directories
    pkg_dir = FindPackageShare('dynamic_obstacle_adapter')
    nav2_bringup_dir = FindPackageShare('nav2_bringup')
    
    # Launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )
    
    map_yaml_arg = DeclareLaunchArgument(
        'map',
        default_value='',
        description='Full path to map yaml file'
    )
    
    nav2_params_file_arg = DeclareLaunchArgument(
        'nav2_params_file',
        default_value='',
        description='Full path to Nav2 parameters file'
    )
    
    use_depth_arg = DeclareLaunchArgument(
        'use_depth',
        default_value='true',
        description='Whether to use depth camera for 3D projection'
    )
    
    # Dynamic obstacle adapter configuration
    adapter_config_file = PathJoinSubstitution([
        pkg_dir, 'config', 'dynamic_obstacle_params.yaml'
    ])
    
    # Dynamic obstacle adapter node
    dynamic_obstacle_node = Node(
        package='dynamic_obstacle_adapter',
        executable='dynamic_obstacle_adapter_node',
        name='dynamic_obstacle_adapter_node',
        parameters=[
            adapter_config_file,
            {
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'use_depth': LaunchConfiguration('use_depth'),
            }
        ],
        output='screen',
        emulate_tty=True,
        respawn=True,
        respawn_delay=2.0
    )
    
    # Nav2 bringup (conditional on parameters being provided)
    nav2_bringup = GroupAction([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                nav2_bringup_dir, '/launch/bringup_launch.py'
            ]),
            launch_arguments={
                'map': LaunchConfiguration('map'),
                'params_file': LaunchConfiguration('nav2_params_file'),
                'use_sim_time': LaunchConfiguration('use_sim_time'),
            }.items()
        )
    ])
    
    # RViz with custom config
    rviz_config_file = PathJoinSubstitution([
        pkg_dir, 'config', 'dynamic_obstacles_rviz.rviz'
    ])
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }],
        output='screen'
    )
    
    return LaunchDescription([
        use_sim_time_arg,
        map_yaml_arg,
        nav2_params_file_arg,
        use_depth_arg,
        dynamic_obstacle_node,
        # Uncomment below to launch Nav2 automatically
        # nav2_bringup,
        # rviz_node,
    ])
