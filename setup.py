"""
Setup script for the Qynthra package.
"""

from setuptools import setup, find_packages

setup(
    name="Qynthra",
    version="0.1.0",
    description="A quantum-enhanced large language model using PennyLane",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Qynthra Contributors",
    author_email="kannanprabakaran84@gmail.com",
    url="https://github.com/organization/Qynthra",
    packages=find_packages(),
    install_requires=[
        "pennylane>=0.30.0",
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.62.0",
        "pillow>=9.0.0",  # For image processing
        "librosa>=0.9.0",  # For audio processing
    ],
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
)