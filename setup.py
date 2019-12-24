from setuptools import setup, find_packages
import os

def requirements(fp: str):
    with open(fp) as f:
        return [r.strip() for r in f.readlines()]

fp_readme = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.rst')
with open(fp_readme, encoding='utf-8') as f:
    long_description = f.read()

setup_requirements = ["pytest-runner", "setuptools_scm"]

setup(
    name='smartmove',
    description=('Utilities for working with biotelemetry and datalogger '
                 'data in Python'),
    long_description=long_description,
    author='Ryan J. Dillon',
    author_email='ryanjamesdillon@gmail.com',
    url='https://github.com/ryanjdillon/smartmove',
    license='MIT',
    packages=find_packages(where="src"),
    package_dir={"":"src"},
    install_requires=requirements("requirements.txt"),
    tests_requires=requirements("requirements_test.txt"),
    include_package_data=True,
    setup_requires=setup_requirements,
    use_scm_version={
        "write_to": "src/smartmove/_version.py",
        "relative_to": __file__,
    },
    keywords=['datalogger','accelerometer','biotelemetry'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.7'],
    zip_safe=True,
    )
