# FreeDeconvolution

## Content

The repository is structured as follows. We only describe the most important files for a new user.
```bash
./
|-- freeDeconvolution: Core of package. 
|  |-- boxes.py    : Functions handling boxes for dichotomy searches on the complex plane
|  |-- core.py     : Core for our method
|  |-- elkaroui.py : Core functions for convex optimization
|  |-- sampling.py : Sampling of empirical measures
|  |-- utils.py    : misc useful functions
|-- ipynb: Contains Python notebooks which demonstrate how the code works
|  |-- DemoFPT.ipynb: Illustrates the measure concentration in FPT.
|  |-- Benchmarks.ipynb: Various useful benchmarks to guide programming choices. For example:
|  |   - Vectorized numpy vs Loop numpy vs Sympy
|  |   - TODO: Contour integral vs Exp convergent scheme (see Trefethen).
|  |-- BranchingPoints.ipynb: Computation of critical points, branching points using the argument principle.
|  |-- Contour.ipynb: Proof of concept for deconvolution using contours then OPRL reconstruction.
|  |-- ElKaroui.ipynb: Python implementation of the method proposed by El Karoui, using convex optimization (cvxpy).
|  |-- Inversions: Exploratory notebook. Future uncertain.
|  |-- Subordination.ipynb: TODO, Not implemented
|-- tests: Unit tests
|-- README.md: This file
```

## Installation

1. Create new virtual environment

```bash
$ python3 -m venv .venv_freeDeconvolution
```

(Do
sudo apt install python3-venv
if needed)

3. Activate virtual environment

```bash
$ source .venv_freeDeconvolution/bin/activate
```

4. Upgrade pip, wheel and setuptools 

```bash
$ pip install --upgrade pip
$ pip install --upgrade setuptools
$ pip install wheel
```

5. Install the `freeDeconvolution` package.

```bash
python setup.py develop
```

6. (Optional) In order to use Jupyter with this virtual environment .venv
```bash
pip install ipykernel
python -m ipykernel install --user --name=.venv_freeDeconvolution
```
(see https://janakiev.com/blog/jupyter-virtual-envs/ for details)

7. (Not needed if step 5 is used) Packages
```bash
pip install numpy matplotlib scipy sympy cvxpy
```

## Configuration
Nothing to do

## Credits
Later