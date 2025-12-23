from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="knapsack-heuristics",
    version="1.0.0",
    author="Paola A. Castillo-Gutiérrez",
    description="A comprehensive system for evaluating greedy heuristics on 0/1 Knapsack Problem instances",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "numba>=0.54.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "pytz>=2021.1",
        "psutil>=5.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
        ],
        "cpuinfo": [
            "py-cpuinfo>=8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "knapsack-experiment=main:main",
        ],
    },
)
