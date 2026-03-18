from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ree_exploration_viz'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        # Include config files
        (os.path.join('share', package_name, 'config'),
            glob(os.path.join('config', '*.rviz'))),
        (os.path.join('share', package_name, 'config'),
            glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='sojavola',
    maintainer_email='sojavolar.2002@gmail.com',
    description='RViz2 Visualization for REE Exploration - Minerals, Obstacles, Heatmaps, Veins',
    license='MIT',
    entry_points={
        'console_scripts': [
            'visualization_node = ree_exploration_viz.visualization_node:main',
            'robot_marker_publisher = ree_exploration_viz.robot_marker_publisher:main',
        ],
    },
)
