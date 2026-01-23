"""
Setup script for Verglas.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="verglas",
    version="0.1.0",
    author="AdalricP",
    description="MusicXML Generator via RLHF Post-Training",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "trl>=0.7.0",
        "peft>=0.6.0",
        "accelerate>=0.24.0",
        "music21>=9.0.0",
        "lxml>=4.9.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "wandb>=0.15.0",
            "tensorboard>=2.14.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "verglas-train=train.py:main",
            "verglas-generate=src.inference.generate:main",
            "verglas-validate=src.inference.validate:main",
        ],
    },
)
