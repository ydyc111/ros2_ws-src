from setuptools import find_packages, setup

package_name = 'arm_gesture_control'

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
    maintainer='smp',
    maintainer_email='smp@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'gesture_arm_node = arm_gesture_control.gesture_arm_node:main',
            'co_gesture_arm_node = arm_gesture_control.co_gesture_arm_node:main',
            'custom_yolo_node = arm_gesture_control.custom_yolo_node:main',          # 新增
            'co_custom_arm_node = arm_gesture_control.co_custom_arm_node:main',      # 新增        
        ],
    },
)
