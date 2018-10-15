from setuptools import setup, find_packages

setup(
    name='gitter',
    version='0.0.1',
    packages=find_packages(),
    url='https://github.com/usajusaj/pitter',
    license='MIT',
    author='Matej Usaj',
    author_email='m.usaj@utoronto.ca',
    description='Python implementation of Gitter by Omar Wagih',
    install_requires=[
        'pandas',
        'scipy>=1.0.0',
        'scikit-image',
        'matplotlib'
    ],
    entry_points={
        'console_scripts': [
            'gitter = gitter.scripts.gitter:main'
        ]
    },
)
