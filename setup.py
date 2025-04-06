#!/usr/bin/env python
"""
Nova2 Setup Script

This setup script defines the package structure and dependencies for the Nova2 project.
"""

from setuptools import setup, find_packages
import os

# Read requirements from requirements.txt if it exists
requirements = []
if os.path.exists('requirements.txt'):
    with open('requirements.txt', 'r') as f:
        requirements = [line.strip() for line in f.readlines() if line.strip()]
else:
    # Core dependencies if requirements.txt is not available
    requirements = [
        'faster-whisper>=0.9.0',
        'numpy>=1.22.0',
        'torch>=2.0.0',
        'sounddevice>=0.4.6',
        'pydub>=0.25.1',
        'PyAudio>=0.2.13',
        'requests>=2.28.0',
        'python-dotenv>=1.0.0',
        'tqdm>=4.65.0',
    ]

setup(
    name="nova2",
    version="0.1.0",
    description="Nova2 - Voice-enabled AI Assistant Framework",
    author="Nova Team",
    author_email="info@nova-ai.org",
    url="https://github.com/nova-ai/nova2",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "nova-server=Nova2.app.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
) 