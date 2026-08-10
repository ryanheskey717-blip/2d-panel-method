# external libraries
import numpy as np
import matplotlib.pyplot as plt

# modules
import Modules.General as General
from Modules.Mesh import Mesh

# inputs
input_file = "Configs/template.yaml"

# load data
config = General.Config(input_file)
mesh = Mesh(config)

# run panel method
mesh.run()

# visualisation
if config.visualisation:
    visualisation = General.Visualisation(config, mesh)
    visualisation.plot()
    # TODO: auto size arrows