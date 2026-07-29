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

# solve etc
mesh.computeAandb()
mesh.solve()

# analysis
velocity_field = General.VelocityField((-0.05, 0.15), (-0.05, 0.05), 100, 50, mesh, config) # (-0.001, 0.001), (-0.001, 0.001)

# plotting
plt.figure()
mesh.plotGeometry(vertices=True, control_points=True, normals=False, tangents=False, vectors_percent_scale=1)
velocity_field.plotStreamlines(colour='velocity')
plt.axis('scaled')
plt.show()