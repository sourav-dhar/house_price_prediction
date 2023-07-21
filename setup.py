from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n','') for req in requirements]
        if '-e .' in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(name = 'HOUSE_PRICE_PRED',
      version = '0.0.3',
      author = 'sourav',
      author_email = 'dharsourav03@gmail.com',
      install_requires = get_requirements('requirements.txt'),
      packages = find_packages())