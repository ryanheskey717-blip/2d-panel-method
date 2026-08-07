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
mesh.calculatePressureOnPanels()
mesh.calculateForcesAndMoments()

# visualisation
if config.visualisation:
    velocity_field = General.VelocityField((-0.02, 0.12), (-0.07, 0.07), 100, 100, mesh, config) # (-0.001, 0.001), (-0.001, 0.001)

    plt.figure()
    mesh.plotGeometry(vertices=False, control_points=True, normals=True, tangents=True, vectors_percent_scale=1)
    velocity_field.plotContour('pressure')
    velocity_field.plotStreamlines(colour='velocity', density=2)
    plt.axis('scaled')
    #plt.show()

    plt.figure()
    plt.plot(mesh.cp.flatten())
    plt.show()