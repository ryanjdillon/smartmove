from setuptools import setup
import os

_here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(_here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='smartmove',
    version='0.1',
    description=('An application to classify seal body density from '
                 'datalogger data using artificial neural networks'),
    long_description=long_description,
    author='Ryan J. Dillon',
    author_email='ryanjamesdillon@gmail.com',
    url='https://github.com/ryanjdillon/smartmove',
    download_url='https://github.com/ryanjdillon/smartmove/archive/0.1.tar.gz',
    license='GPL-3.0+',
    packages=['smartmove'],
    install_requires=[
        'scipy',
        'seaborn',
        'sklearn',
        'theanets',
        'theano',
        'pandas',
        'pylleo',
        'pyotelem',
        'yamlord',
        ],
    include_package_data=True,
    keywords=['datalogger','accelerometer','biotelemetry'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.5'],
    zip_safe=False,
    )
