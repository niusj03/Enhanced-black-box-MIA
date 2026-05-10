# coding:utf-8

from setuptools import setup, find_packages

setup(
    name="simmia",
    version="1.0.0",
    description="A membership inference toolkit on LLMs.",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires="~=3.12",
    install_requires=[
        "black~=25.1.0",
        "sentence-transformers>=3.3.0",
        "transformers>=4.52.0",
        "huggingface_hub~=0.36.0",
        "datasets~=2.21.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.3.0",
        "nltk>=3.8.0",
        "tqdm>=4.65.0",
        "torch>=2.0.0",
        "accelerate>=0.20.0",
    ],
    extras_require={
        "api": [
            "openai~=2.11.0",
            "google-genai~=1.55.0",
            "anthropic~=0.75.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "simmia.benchmark = simmia.run:cli_main",
        ],
    },
)
