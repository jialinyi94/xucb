from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "ucb_cython",
        ["ucb_cython.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=["-O3"]  # Add optimization flag
    )
]

setup(
    name='ucb_algorithm',
    version='0.2.1',
    description='Upper Confidence Bound (UCB) Algorithm Implementation using Cython and NumPy',
    ext_modules=cythonize(extensions, annotate=True),  # Enable annotation for optimization hints
    install_requires=['numpy'],
)