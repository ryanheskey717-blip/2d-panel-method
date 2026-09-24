# import modules
import Modules.PanelMethod as pm

# initialise
# eg. single case
#case = pm.SingleCase("Configs/template.yaml")

# eg. sweep over alpha
case = pm.Sweep("Configs/template.yaml", "Configs/sweep_template.yaml")

# run
case.run()