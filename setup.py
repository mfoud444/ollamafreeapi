from setuptools import setup, find_packages
import os

# Read version from version.py without importing
def get_version():
    version_file = os.path.join('ollamafreeapi', 'version.py')
    with open(version_file, 'r') as f:
        for line in f:
            if line.startswith('VERSION'):
                return line.split('=')[1].strip().strip('"\'')
    return '0.0.0'

setup(
    name="ollamafreeapi",
    version=get_version(),
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