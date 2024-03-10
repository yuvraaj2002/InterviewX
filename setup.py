from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> List[str]:
    """
     This function will take file path as input and return list of libraries to install
    :param file_path:
    :return: list
    """

    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
    name="InterviewX",
    version="0.0.1",
    author="Yuvraj Singh",
    author_email="ys2002github@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
