from setuptools import find_packages, setup

setup(
    name='pysitcom',
    packages=find_packages(include=['pysitcom']),
    author_email="bartlomiej-kizielewicz@zut.edu.pl",
    version='0.1.0',
    description='Stochastic optimization techniques in identify the multi-criteria decision model',
    author='Bartlomiej Kizielewicz',
    license='MIT',
    install_requires = [
        "numpy",
        "pyswarms",
        "scikit-learn==0.22",
        "mlrose",
        "pyyaml",
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)