import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'mujoco_ros'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'model'), glob('model/*.xml')),
        (os.path.join('share', package_name, 'model', 'meshes'), glob('model/meshes/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bicarcom',
    maintainer_email='ktu512693@gmail.com',
    description='MG400 MuJoCo simulation ROS2 control package',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'mujoco_controller = mujoco_ros.mujoco_controller:main',
            'command_sender = mujoco_ros.command_sender:main',
        ],
    },
)
