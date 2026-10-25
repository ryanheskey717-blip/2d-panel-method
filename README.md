# 2D Panel Method

## Setup

To organise the required packages nicely, a virtual environment is recommended, but not strictly required. The included tests make sure the code is working correctly.

### Create and setup virtual environment

#### Mac:

```
python3 -m venv venv
source venv/bin/activate
```

```
pip install -r requirements.txt
```

#### Windows:

##### powershell:

```
python -m venv venv
venv\Scripts\activate.ps1
```

##### cmd:

```
python -m venv venv
venv\Scripts\Activate.bat
```

```
pip install -r requirements.txt
```

#### deactivate venv

```
deactivate
```

### Run Tests

Tests are conducted using the library ```pytest```. To test the code, simply run in the main folder:

```
pytest
```

## Running the Code

### Geometry

Geometry is defined by a set of coordinates in ```.dat``` or ```.txt``` files. These should have the form of:

```
NACA 0012 Airfoil
  1.000000 -0.000000
  0.975528  0.003501
  0.904508  0.013071
  0.793893  0.026405
  etc.
```

The first line can be used for a description and is ignored when loaded. The first and last coordinates do not necessarily have to be the same, but large gaps could cause issues in the code. These files can be written by hand, or [Airfoil Tools](http://airfoiltools.com/airfoil/naca4digit?MNaca4DigitForm%5Bcamber%5D) and [UIUC Database](https://m-selig.ae.illinois.edu/ads/coord_database.html#N) can be used to generate coordinates for many different aerofoil already in this form. These outputs can be copy and pasted into a ```.dat``` or ```.txt``` file. Code is set up so that geometry files are placed in the Geometries folder, but the actual path is defined in the input file. 

### Input Files

#### Main inputs

Input files are written in yaml, and have a specific form. An example of one of these (```template.yaml```) is included in the Configs directory. In this file, all relevant settings and parameters can be defined, with guides of what is supported included in the template file. I would recommend copying this and modifying to suit your needs. The settings in these files are loaded into a Config class when the panel method is initialised. This is the only input required when a single case is being ran. 

#### Sweep inputs

To set up a sweep, which at the moment can only be done over a range of angles of attack, a sweep config needs to be provided *as well as* a regular input file. The regular input file contains all the settings used by the solver, and the sweep file tells the code how to modify the input conditions for each case. An example of one of these is also included in the Configs directory, showing how they are used. 

### Running the panel method

It is a very simple process to run this code. ```main.py``` shows some sample code of how to run the template input files. To begin with, the required modules must be imported with:
```
import Modules.PanelMethod as pm
```
Next is to initialise the mesh and solver, this done in one line with:
```
case = pm.SingleCase("Configs/template.yaml")
```
or
```
case = pm.Sweep("Configs/template.yaml", "Configs/sweep_template.yaml")
```
for sweeps.

To run the actual solver, simply write:
```
case.run()
```
Which will run whatever method has been setup. 

Any outputs or even internal parameters can be accessed with regular class notation, example:
```
print('Lift Coefficient:', case.mesh.c_lift)
print('Drag Coefficient:', case.mesh.c_drag)
```
**Note:** sweeps overwrite the case variable on each new case to avoid memory issues so this method of accessing variable will results in those from the very last case being accessed. This is why sweeps are forced to save important parameters to a ```.csv``` file in Output.

### Results Output

If the ```write_to_file``` setting in the input file is set to ```True```, then a ```.csv``` file of many aerodynamic forces, moments and coefficients will be saved in the Output directiory. The name of this can be specified in the input file. 

### Geometry and Flow Visualisation

The code contains support for visualising the geometry and flow solution for a single case. This is toggled on and off in the input file, but is forced to off for sweeps. The process of calculating off-body velocities can take significantly longer than the actual solving. The visualisation can show different parts of the geometry including control points, and normal and tangenetial vectors. The global coordinate system van be plotted. On the fluid side, stream lines can be plotted as well as contours of the velocity components, magnitude or pressure. This is all controlled inside the input file.

### Analysis of Sweep Results

To conduct post processing/analysis, import the following module:
```
import Modules.SweepAnalysis as swan
```
This module contains support for loading a results ```.csv``` file and plotting chosen parameters over angle of attack. Any header included in the ```.csv``` can be used to choose what to plot. This module can also create subplots of multiple parameters using a list of header names. This module has support to load an XFOIL results file from [Airfoil Tools](http://airfoiltools.com/polar/details?polar=xf-n0012-il-200000-n5), and compare to the generated results by displaying on the same plot. The code below does just this with some provided sample data:
```
results = swan.Results('Output/naca0012_coefficients_bl.csv')
comp_res = swan.XFOILResults('AirfoilData/naca0012_Re_200000.txt')
results.subplots(x=['alpha_deg', 'alpha_deg', 'alpha_deg'], y=['c_l', 'c_d', 'c_m'], nrows=1, ncols=3, compare=comp_res)
```

This is a brief overview of how to use this code, and documentation for how the internals work can be found in the ```docs``` directory.


## Future Improvements:
This code is still a work in progress, and there are many parts that are not working correctly or are not realistic. Below are some future improvements that will hopefully be made, which will increase the quality of results.

- results with viscous coupling jump around, this is likely due to a horrible stagnation point calculation method.
- reformat how the viscous coupling is turned on and off in the code.
- add turbulent blasius boundary layer (at least).
- maybe add more complex boundary layer (that includes pressure gradients).
- add boundary layer to velocity field calculation.