from setuptools import setup, find_packages
from glob import glob

package_name = "rl_agent_ppo"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Mariem Ameni",
    maintainer_email="user@example.com",
    description="PPO (continuous actions) lane-following agent for Duckiebot in Gazebo Harmonic (ROS 2 Jazzy)",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "ppo_training_node = rl_agent_ppo.ppo_training_node:main",
            "diag_node         = rl_agent_ppo.diag_node:main",
        ],
    },
)
