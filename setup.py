from setuptools import find_packages, setup

setup(
    name="trajectolab",
    version="0.1.0",
    description="Optimal control framework using Radau Pseudospectral Method",
    author="TrajectoLab Authors",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.18.0",
        "matplotlib>=3.1.0",
        "scipy>=1.4.0",
        "casadi>=3.5.0",  # CasADi is used for the optimization backend
        "pandas>=1.0.0",  # Used for data export
    ],
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    keywords="optimal control, trajectory optimization, collocation, pseudospectral methods",
)
