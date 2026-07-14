from setuptools import find_packages, setup

package_name = 'realworld_ros'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bicarcom',
    maintainer_email='ktu512693@gmail.com',
    description='Dobot MG400 real robot ROS2 control package',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'dobot_control = realworld_ros.dobot_control:main',
            'point_cycler = realworld_ros.point_cycler:main',
        ],
    },
)
