# import modules
import Modules.PanelMethod as pm
import Modules.SweepAnalysis as swan

# initialise
# eg. single case
case = pm.SingleCase("Configs/template.yaml")

# eg. sweep over alpha
#case = pm.Sweep("Configs/template.yaml", "Configs/sweep_template.yaml")

# run
case.run()
print('lift:', case.mesh.c_lift)
print('drag:', case.mesh.c_drag)
print('mome:', case.mesh.c_moment)

# create plots
results = swan.Results('Output/naca0012_coefficients_bl.csv')
comp_res = swan.XFOILResults('AirfoilData/naca0012_Re_200000.txt')
results.subplots(x=['alpha_deg', 'alpha_deg', 'alpha_deg'], y=['c_l', 'c_d', 'c_m'], nrows=1, ncols=3, compare=comp_res)