from setuptools import find_packages, setup

package_name = 'mp_gesture_pose'

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
            'mp_node = mp_gesture_pose.mp_node:main',
            'mp_node_action = mp_gesture_pose.mp_node_action:main',
        ],
    },
)
