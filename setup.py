#!/usr/bin/env python

from setuptools import setup

setup(
    name="Financial Crises Predictor",
    version="0.1",
    description="TBD",
    author="Arushi Sinha",
    packages=["src"],
    install_requires=[
        "pandas",
        "numpy",
        "bayesian-optimization",
        "datasets",
        "fancyimpute",
        "huggingface_hub",
        "imbalanced-learn",
        "knnimpute",
        "lightgbm",
        "matplotlib",
        "openpyxl",
        "python-dotenv",
        "requests",
        "seaborn",
        "scikit-learn",
        "statsmodels",
        "torch",
        "tqdm",
        "umap-learn",
        "xgboost",
        "world_bank_data",
        "yfinance",
    ],
)