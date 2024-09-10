#! /usr/bin/env python

"""Installer script."""
import setuptools

with open("README.md") as f:
    long_description = f.read()

setuptools.setup(
    name="cmcc-reactions",
    version="0.0.1",
    description="Tools for working with mechanochemically activated reactions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Center-for-Mechanical-Control-of-Chem/cmcc_reactions",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "ase",
        "rdkit",
        "torch",
        "torch_geometric",
        "matplotlib",
        "git+https://github.com/isayevlab/AIMNet2",
        "mccoygroup-mcutils",
        "nglview"
    ]
)
