# import modules
import Modules.PanelMethod as pm
import Modules.SweepAnalysis as san

# initialise
# eg. single case
#case = pm.SingleCase("Configs/template.yaml")

# eg. sweep over alpha
case = pm.Sweep("Configs/template.yaml", "Configs/sweep_template.yaml")

# run
case.run()

# create plots
results = san.Results('Output/test_coefficients.csv')
results.plot(x='alpha_deg', y='c_l')
results.plot(x='alpha_deg', y='c_d')
results.plot(x='alpha_deg', y='c_m')