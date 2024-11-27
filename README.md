# LIME with Labelme

### Anaconda

You need install [Anaconda](https://www.continuum.io/downloads), then run below:

```bash
# python3
conda env create --name envname --file=environments.yml
conda activate lime_labelme
```

## Usage
Before running the following command, 
first copy the file from .../labelme/labelme/__main__.py to .../labelme/__main__.py path. Then in .../labelme/__main__.py path to run the

```bash
python3 __main__.py
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
