#!/usr/bin/env python3

from catkin_pkg.python_setup import generate_distutils_setup
from setuptools import setup


setup_args = generate_distutils_setup(
    packages=["stage_cartesian_trajectory"],
    package_dir={"": "src"},
)

setup(**setup_args)
