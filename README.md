# LIME with Labelme

## Preparing the runtime environment
There are two ways to prepare a runtime environment, and the second is the recommended python virtual environment.
### Anaconda

You need install [Anaconda](https://www.continuum.io/downloads), then run below:

```bash
# python3
conda env create --name envname --file=environments.yml
conda activate lime_labelme
```

### Python venv
This project can be run using python virtual environment (venv).
1. Create a Python virtual environment
```bash
python -m venv labelme_lime  # Make sure the Pyhton version is 3.10
```
2. Activate the virtual environment

Windows (CMD/PowerShell):
```bash 
labelme_lime\Scripts\activate
```
macOS/Linux (bash/zsh):
```bash 
source labelme_lime/bin/activate
```

3. Install requirements.txt dependencies

```bash
pip install -r requirements.txt
```

## Usage

Before running the following command,
first copy the file from .../labelme/labelme/__main__.py to .../labelme/__main__.py path. Then in .../labelme/__main__.py path to run the

```bash
python __main__.py
```

### How to build standalone executable

Below shows how to build the standalone executable on macOS, Linux and Windows.

```bash
# Setup conda
conda create --name labelme python=3.9
conda activate labelme

# Build the standalone executable
pip install .
pip install 'matplotlib<3.3'
pip install pyinstaller
pyinstaller labelme.spec
dist/labelme --version
```
