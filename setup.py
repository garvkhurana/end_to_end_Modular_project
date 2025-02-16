from setuptools import find_packages, setup
from typing import List


hypen="-e ."

def get_requirements(path:str)->List[str]:
    requirements=[]
    with open(path) as file_obj:
        for line in file_obj:
            requirements=file_obj.readlines()
            requirements=[req.replace("\n","") for req in requirements]
            
            
            if hypen in requirements:
                requirements.remove(hypen)
                
    return requirements

            
                
    

setup(
name='ml_project', 
version='0.0.1',
author="garv",
author_email="garvkhurana1234567@gmail.com",
packages=find_packages(),
install_requires=get_requirements('requirements.txt')    
)