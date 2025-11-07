"""
Setup script for AI Sleep Constructs package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="ai-sleep",
    version="0.1.0",
    author="Tionne Smith",
    author_email="",
    description="Production-ready framework for engineered sleep cycles in language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/electricwolfemarshmallowhypertext/ai-sleep",
    project_urls={
        "Bug Reports": "https://github.com/electricwolfemarshmallowhypertext/ai-sleep/issues",
        "Source": "https://github.com/electricwolfemarshmallowhypertext/ai-sleep",
        "DOI": "https://doi.org/10.5281/zenodo.17547016"
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "huggingface": ["transformers>=4.20.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    license="CC-BY-NC-SA-4.0",
    keywords="language-models ai machine-learning optimization sleep-cycles performance-monitoring",
)
