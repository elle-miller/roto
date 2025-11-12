"""Installation script for the 'isaaclab_rl' python package."""

import os

from setuptools import setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # NOTE: Add dependencies
]


# Installation operation
setup(
    name="roto",
    packages=["roto"],
    author="Elle Miller",
    maintainer="Elle Miller",
    url="https://github.com/elle-miller/roto",
    version="1.0.0",
    description="project",
    keywords=["roto"],
    install_requires=INSTALL_REQUIRES,
    license="BSD-3-Clause",
    include_package_data=True,
    # python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Isaac Sim :: 5.1.0",
    ],
    zip_safe=False,
)
