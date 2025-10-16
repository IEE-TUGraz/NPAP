from setuptools import setup, find_packages

setup(
    name="npap",
    version="0.1.0",
    author="Marco Antonio Arnaiz Montero",
    description="Network Partitioning & Aggregation Package",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "networkx",
        "numpy",
        "pandas",
        "plotly",
        "scikit-learn",
        "scikit-learn-extra",
    ],
)
