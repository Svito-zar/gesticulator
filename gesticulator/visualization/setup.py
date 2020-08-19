# Always prefer setuptools over distutils
from setuptools import setup, find_packages

setup(
    name="motion_visualizer",
    version="0.0.1",
    packages=["motion_visualizer", "pymo"],
    install_requires=[
        "matplotlib",
        "scipy",
        "pyquaternion",
        "pandas",
        "sklearn",
        "transforms3d",
        "bvh",
    ],
    package_data={"motion_visualizer": ["data/data_pipe.sav"]},
    package_dir={"motion_visualizer": "motion_visualizer"},
)
