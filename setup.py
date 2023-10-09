#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages

setup(
    name="churn-prediction",
    version="1.0",
    description="Churn prediction with LSTM and embeddings",
    author="Yanina Kutovaya",
    author_email="kutovaiayp@yandex.ru",
    url="https://github.com/YaninaK/churn-prediction.git",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)