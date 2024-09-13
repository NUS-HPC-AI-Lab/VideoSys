from typing import List

from setuptools import find_packages, setup
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info
from setuptools.command.install import install


def fetch_requirements(path) -> List[str]:
    """
    This function reads the requirements file.

    Args:
        path (str): the path to the requirements file.

    Returns:
        The lines in the requirements file.
    """
    with open(path, "r") as fd:
        requirements = [r.strip() for r in fd.readlines()]
        requirements.remove("colossalai")
        return requirements


def fetch_readme() -> str:
    """
    This function reads the README.md file in the current directory.

    Returns:
        The lines in the README file.
    """
    with open("README.md", encoding="utf-8") as f:
        return f.read()


def custom_install():
    return ["pip", "install", "colossalai", "--no-deps"]


class CustomInstallCommand(install):
    def run(self):
        install.run(self)
        self.spawn(custom_install())


class CustomDevelopCommand(develop):
    def run(self):
        develop.run(self)
        self.spawn(custom_install())


class CustomEggInfoCommand(egg_info):
    def run(self):
        egg_info.run(self)
        self.spawn(custom_install())


setup(
    name="videosys",
    version="2.0.0",
    packages=find_packages(
        exclude=(
            "videos",
            "tests",
            "figure",
            "*.egg-info",
        )
    ),
    description="VideoSys",
    long_description=fetch_readme(),
    long_description_content_type="text/markdown",
    license="Apache Software License 2.0",
    install_requires=fetch_requirements("requirements.txt"),
    python_requires=">=3.7",
    cmdclass={
        "install": CustomInstallCommand,
        "develop": CustomDevelopCommand,
        "egg_info": CustomEggInfoCommand,
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Environment :: GPU :: NVIDIA CUDA",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
    ],
)
