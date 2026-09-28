from setuptools import setup

package_name = 'tf_tools'

setup(
    name=package_name,
    version='0.0.0',
    packages=['tf_tools'],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='TF tools',
    license='Apache License 2.0',
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/tf_tools']),
        ('share/tf_tools', ['package.xml']),
    ],
    entry_points={
        'console_scripts': [
            'odom_tf = tf_tools.odom_tf:main',
        ],
    },
)