#!/usr/bin/env python3

from catkin_pkg.python_setup import generate_distutils_setup
from setuptools import setup


setup(
    **generate_distutils_setup(
        packages=["stage_constraint_planner"],
        package_dir={"": "src"},
    )
)
