# setup.py file for the TFData package
from setuptools import setup

setup(
    name='IsoDat2D',
    version='1.0.0',
    description='To process thin film X-ray total scattering data using unsupervised machine learning',
    long_description= 'Combining non-negative matrix factorization and clustering algorithms to process thin film X-ray total scattering data and separate isotropic and anisotropic components',
    author_email='dalverson@ufl.edu',
    packages = ['IsoDAT2D',],
    url='https://github.com/dnalverson/IsoDAT2D',
    install_requires = ['matplotlib', 'pyFAI', 'scipy', 'numpy', 'scikit-learn']
)