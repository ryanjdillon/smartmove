from setuptools import setup, find_packages
import os

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='smartmove',
    version='0.1',
    description=('Utilities for working with biotelemetry and datalogger '
                 'data in Python'),
    long_description=long_description,
    author='Ryan J. Dillon',
    author_email='ryanjamesdillon@gmail.com',
    url='https://github.com/ryanjdillon/smartmove',
    download_url='https://github.com/ryanjdillon/smartmove/archive/0.1.tar.gz',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'scipy==0.19.1',
        'seaborn==0.8',
        'scikit-learn==0.18.2',
        'theanets==0.7.3',
        'theano==0.9.0',
        'pandas==0.20.3',
        'gsw==3.1.1',
        'numpy==1.13.1',
        'matplotlib==2.0.2',
        'yamlord==0.4',
        'pylleo==0.4',
        'pyotelem==0.4',],
    include_package_data=True,
    keywords=['datalogger','accelerometer','biotelemetry'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.5'],
    zip_safe=False,
    )
