# 2d-panel-method

# Work in Progress

## TODO:

- add turbulent blasius boundary layer
- maybe add more complex boundary layer (that includes pressure gradients) if i have time
- add outputs to file
- format each function correctly
- create documentation

## Create venv

### Mac:

```
python3 -m venv venv
source venv/bin/activate
```

```
pip install -r requirements.txt
```

### Windows:

#### powershell:

```
python -m venv venv
venv\Scripts\activate.ps1
```

#### cmd:

```
python -m venv venv
venv\Scripts\Activate.bat
```

```
pip install -r requirements.txt
```

### deactivate venv

```
deactivate
```

## airfoil generator:

http://airfoiltools.com/airfoil/naca4digit?MNaca4DigitForm%5Bcamber%5D

then copy and paste .data file into a new .dat or .txt file in Geometries
