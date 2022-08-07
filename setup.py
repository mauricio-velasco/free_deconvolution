from setuptools import Extension, setup, find_packages
from os import path

local_path = path.abspath(path.dirname(__file__))

print("Local path: ", local_path)
print("")

print("Launching setup...")
# Setup
setup(
    name='freeDeconvolution',

    version='0.01',

    description='Computational solutions to Free Deconvolution',
    long_description=""" Computational solutions to Free Deconvolution.
    In this module, we implement and benchmark various computational methods 
    for computing a free deconvolution.
    - Convex optimization following El Karoui
    - Subordination and classical deconvolution following Arizmendi, Tarrago, Vargas
    - Complex methods following Chhaibi, Gamboa, Kammoun, Velasco.
    """,
    url='',

    author='Chhaibi, Gamboa, Kammoun, Velasco',
    author_email='Anonymous',

    license='MIT License',

    install_requires=["numpy", "matplotlib", "scipy", "sympy", "cvxpy"],

    keywords='',

    packages=find_packages(),

    entry_points={
        'console_scripts': [
            'sample=sample:main',
        ],
    },

    ext_modules=[],
)