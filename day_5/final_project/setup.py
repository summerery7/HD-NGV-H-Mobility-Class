from setuptools import find_packages, setup

package_name = 'final_project'

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
    description='YOLO + Camera + Robot + ROS2 integration project skeleton',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'robot_control = final_project.robot_control:main',
            'yolo_detector = final_project.yolo_detector:main',
            'camera_calibration = final_project.camera_calibration:main',
            'main_controller = final_project.main_controller:main',
        ],
    },
)
