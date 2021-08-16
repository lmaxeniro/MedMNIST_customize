from setuptools import setup, find_packages

import medmnist


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


def requirements():
    with open('requirements.txt') as f:
        required = f.read().splitlines()
    return required


setup(
    name='medmnist',
    version=medmnist.info.__version__,
    url=medmnist.info.HOMEPAGE,
    license='Apache-2.0 License',
    author='Jiancheng Yang and Rui Shi',
    author_email='jekyll4168@sjtu.edu.cn',
    description='MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark for Medical Image Analysis',
    long_description=readme(),
    packages=find_packages(),
    install_requires=requirements(),
    zip_safe=True
)