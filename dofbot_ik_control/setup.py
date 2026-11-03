from setuptools import find_packages, setup

package_name = 'dofbot_ik_control'

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
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ik_arm_node = dofbot_ik_control.ik_arm_node:main',
            'ik_arm_grab_node = dofbot_ik_control.ik_arm_grab_node:main',
            'gamepad_node = dofbot_ik_control.gamepad_node:main',
            'ik_arm_gamepad_node = dofbot_ik_control.ik_arm_gamepad_node:main',
        ],
    },
)
