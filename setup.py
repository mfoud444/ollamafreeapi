from setuptools import setup, find_packages
from ollamafreeapi.version import VERSION

setup(
    name="ollamafreeapi",
    version=VERSION,
    packages=find_packages(),
    package_data={
        'ollamafreeapi': ['ollama_json/*.json'],
    },
    install_requires=[
        'ollama>=0.1.0',
    ],
    author="Mohammed Foud",
    author_email="mfoud444",
    description="A lightweight client for interacting with LLMs served via Ollama",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/mfoud444/ollamafreeapi",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)