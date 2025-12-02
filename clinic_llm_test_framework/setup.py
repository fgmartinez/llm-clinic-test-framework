"""Package installer for clinic_llm_test_framework.

This setup file allows the framework to be installed in editable mode
(``pip install -e .``) so that tests and scripts can import its
modules directly.  It lists the package metadata and dependencies.
"""

from setuptools import setup, find_packages


setup(
    name="clinic_llm_test_framework",
    version="0.1.0",
    description="Evaluation framework for a medical clinic virtual assistant using LLMs",
    packages=find_packages(),
    install_requires=[
        "deepeval",
        "langchain-openai",
        "langchain-google-genai",
        "jinja2",
        "scikit-learn",
        "numpy",
        "python-dotenv",
    ],
)