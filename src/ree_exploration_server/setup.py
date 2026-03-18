from setuptools import find_packages, setup

package_name = 'ree_exploration_server'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sojavola',
    maintainer_email='sojavolar.2002@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
#        'test': [
#            'pytest',
#        ],

    },
    entry_points={
        'console_scripts': [
            'server_node = ree_exploration_server.server_node:main',
        ],
    },
)
