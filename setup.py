from setuptools import setup, find_packages
from glob import glob

setup(
    name='remove_watermark',
    version='1.0',
    packages=find_packages(),
    package_data={'': glob('**/*', recursive=True) + glob('.**/*', recursive=True)},
    include_package_data=True,
)

