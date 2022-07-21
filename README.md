# FreeDeconvolution

## Content

The repository is structured as follows. We only describe the most important files for a new user.
```bash
./
|-- freeDeconvolution: Core of package. 
|-- ipynb: Contains Python notebooks which demonstrate how the code works
|  |-- DemoFPT.ipynb: Illustrates the measure concentration in FPT.
|  |-- Inversion.ipynb: Current development file.
|-- tests: Unit tests
|-- README.md: This file
```

## Installation

1. Create new virtual environment

```bash
$ python3 -m venv .venv_freeDeconv
```

(Do
sudo apt install python3-venv
if needed)

3. Activate virtual environment

```bash
$ source .venv_freeDeconv/bin/activate
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
python -m ipykernel install --user --name=.venv_freeDeconv
```
(see https://janakiev.com/blog/jupyter-virtual-envs/ for details)

## Configuration
Nothing to do

## Credits
Later