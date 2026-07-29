# general classes and functions
# (loading / printing / visualising)

import yaml
import numpy as np
import matplotlib.pyplot as plt

# ==================== Loading =================== #

# load and store config
class Config:
    def __init__(self, input_file):

        # load yaml file
        with open(input_file, "r") as file:
            self.full_data = yaml.safe_load(file)

        # settings
        self.verbose = self.full_data["verbose"]

        # input
        self.geo_file = self.full_data["geo_file"]
        self.chord = self.full_data["chord"]

        # conditions
        self.M_inf = self.full_data["M_inf"]
        self.T_inf = self.full_data["T_inf"]
        self.alpha = np.deg2rad(self.full_data["alpha"])

        # solver
        self.control_point_offset = float(self.full_data["control_point_offset"])

        # calculate other important values
        self.V_inf = self.M_inf * np.sqrt(1.4 * 287 * self.T_inf)
        self.V_inf_vec = np.array([
            self.V_inf * np.cos(self.alpha),
            self.V_inf * np.sin(self.alpha)
        ])

# ==================== Analysing ========================= #

class VelocityField:
    def __init__(self, xlim, ylim, nx, ny, geo_mesh, config):

        # get x and y coords
        x = np.linspace(xlim[0], xlim[1], nx)
        y = np.linspace(ylim[0], ylim[1], ny)

        # get grid
        self.meshgrid = np.meshgrid(x, y)

        # calculate velocity at each grid poin
        # initialise with freestream velocity
        self.v = np.ones_like(self.meshgrid)
        self.v[0] *= config.V_inf_vec[0]
        self.v[1] *= config.V_inf_vec[1]

        # add in geometry influence
        for i in range(len(self.meshgrid[0][0, :])):
            for j in range(len(self.meshgrid[0][:, 0])):
                u, v = geo_mesh.getConstantSourcePanelsVelocity((self.meshgrid[0][j, i], self.meshgrid[1][j, i]))
                self.v[0][j, i] += u
                self.v[1][j, i] += v




    # ================ Velocity Plotting ================== #

    def plotStreamlines(self, colour='C0', density=1):
        if colour == 'velocity':
            colour = np.sqrt(self.v[0]**2 + self.v[1]**2)

        # plot streamlines
        strm = plt.streamplot(self.meshgrid[0], self.meshgrid[1], self.v[0], self.v[1], color=colour, linewidth=1, density=density, cmap='viridis')
        plt.colorbar(strm.lines)

        # remove arrows
        ax = plt.gca()
        for art in ax.get_children():
            if isinstance(art, plt.matplotlib.patches.FancyArrowPatch):
                art.remove()

    def plotContour(self, var, colorbar=True):
        if var == "vx":
            contour = plt.contourf(self.meshgrid[0], self.meshgrid[1], self.v[0], levels=50, cmap='viridis')
        elif var == "vy":
            contour = plt.contourf(self.meshgrid[0], self.meshgrid[1], self.v[1], levels=50, cmap='viridis')
        elif var == "mag":
            contour = plt.contourf(self.meshgrid[0], self.meshgrid[1], np.sqrt(self.v[0]**2 + self.v[1]**2), levels=50, cmap='viridis')
        elif var == "pressure":
            print("Not yet supported: Check back soon!")
            return
        else:
            print("\nWarning!: No valid variable selected for contour plot.\n")
            return
        
        if colorbar:
            plt.colorbar(contour, label=var)