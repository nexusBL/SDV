from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'sdv_perception'

setup(
    name=package_name,
    version='2.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml') + glob('../../config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nvidia',
    maintainer_email='nvidia@sdv.local',
    description='SDV Perception Pipeline — YOLOv8 + Lane + LiDAR + Depth Fusion for QCar2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'perception_node = sdv_perception.perception_node:main',
            'camera_bridge = sdv_perception.camera_bridge:main',
        ],
    },
)
