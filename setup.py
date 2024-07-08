from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "ucb.ucb_wrapper",
        ["src/ucb/ucb_wrapper.pyx", "src/ucb/ucb.cpp"],
        include_dirs=[np.get_include()],
        language="c++"
    ),
]

setup(
    name="ucb-algorithm",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A fast Upper Confidence Bound algorithm",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ucb-package",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.18.0",
    ],
    ext_modules=cythonize(extensions),
    zip_safe=False,
)
