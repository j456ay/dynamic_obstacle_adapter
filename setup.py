from setuptools import setup
import os
from glob import glob

package_name = 'dynamic_obstacle_adapter'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), 
         glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), 
         glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Robot Developer',
    maintainer_email='robot@example.com',
    description='ROS2 package for converting YOLO detections to dynamic obstacles for Nav2',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dynamic_obstacle_adapter_node = dynamic_obstacle_adapter.dynamic_obstacle_adapter_node:main',
        ],
    },
)
