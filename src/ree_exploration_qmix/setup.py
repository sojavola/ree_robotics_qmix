from setuptools import setup
import os
from glob import glob

package_name = 'ree_exploration_qmix'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sojavola',
    maintainer_email='sojavola@gmail.com',
    description='QMIX for Rare Earth Elements Exploration with ROS2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'qmix_trainer = ree_exploration_qmix.qmix_trainer_node:main',
            'qmix_agent = ree_exploration_qmix.qmix_agent_node:main',
        ],
    },
)