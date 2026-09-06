# general classes and functions
# (loading / printing / visualising)

import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

import Modules.ElementaryFlows as flows

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
        self.ref_position = np.array(self.full_data["ref_position"])

        # conditions
        self.M_inf = self.full_data["M_inf"]
        self.T_inf = self.full_data["T_inf"]
        self.rho_inf = self.full_data["rho_inf"]
        self.mu_inf = self.full_data["mu_inf"]
        self.nu_inf = self.mu_inf / self.rho_inf
        self.p_inf = self.full_data["p_inf"]

        self.alpha = np.deg2rad(self.full_data["alpha"])

        # solver
        self.control_point_offset = float(self.full_data["control_point_offset"])
        self.convergence_criterion = float(self.full_data["convergence_criterion"])
        if self.convergence_criterion == 0.0: # single iteration
            self.run_till_converged = False
        else: # let run until convergence
            self.run_till_converged = True
        
        # turbulence/viscous
        self.inflation_layers = self.full_data["inflation_layers"]
        self.wake_starting_index = int(self.full_data["wake_starting_index"])
        self.wake_points = int(self.full_data["wake_points"])
        self.wake_length = float(self.full_data["wake_length"])

        # flow properties
        self.pressure_calculation = self.full_data["pressure_calculation"]

        # visualisation
        self.visualisation = self.full_data["visualisation"]
        self.scale = self.full_data["scale"]

        # output
        self.write_to_file = self.full_data["output"]["write_to_file"]
        self.out_coeff_file = self.full_data["output"]["coefficients"]

        # calculate other important values
        self.V_inf = self.M_inf * np.sqrt(1.4 * 287 * (self.T_inf + 273.15))
        self.V_inf_vec = np.array([
            self.V_inf * np.cos(self.alpha),
            self.V_inf * np.sin(self.alpha)
        ])
        self.Re = self.V_inf * self.chord * self.rho_inf / self.mu_inf

# ==================== Analysing ========================= #

class VelocityField:
    def __init__(self, geo_mesh, config, area_type, xlim=None, ylim=None, nx=None, ny=None):

        # input values
        self.config = config
        self.area_type = area_type

        # create mesh
        if self.area_type == 'cartesian':
            # get coords
            x = np.linspace(xlim[0], xlim[1], nx)
            y = np.linspace(ylim[0], ylim[1], ny)

            # get grid
            self.meshgrid = np.meshgrid(x, y)

        elif self.area_type == 'conformal':
            # spacing
            eta = np.geomspace(1, 10, ny)
            eta = (eta - 1) / (10 - 1) * ylim

            # get coords
            x = np.zeros((len(geo_mesh.control_points), ny))
            y = np.zeros((len(geo_mesh.control_points), ny))
            for j in range(ny):
                x[:, j] = geo_mesh.control_points[:, 0] + geo_mesh.normals[:, 0] * eta[j]
                y[:, j] = geo_mesh.control_points[:, 1] + geo_mesh.normals[:, 1] * eta[j]
            
            # add in grid coming from the wake (using vertices not middle of panels)
            x_wake1 = np.zeros((len(geo_mesh.wake_vertices), ny))
            y_wake1 = np.zeros((len(geo_mesh.wake_vertices), ny))
            x_wake2 = np.zeros((len(geo_mesh.wake_vertices), ny))
            y_wake2 = np.zeros((len(geo_mesh.wake_vertices), ny))
            for j in range(ny): # gets projected outwards with same normal as last panel on each side
                x_wake1[:, j] = geo_mesh.wake_vertices[:, 0][::-1] + geo_mesh.normals[0, 0] * eta[j]
                y_wake1[:, j] = geo_mesh.wake_vertices[:, 1] + geo_mesh.normals[0, 1] * eta[j]
                x_wake2[:, j] = geo_mesh.wake_vertices[:, 0] + geo_mesh.normals[-1, 0] * eta[j]
                y_wake2[:, j] = geo_mesh.wake_vertices[:, 1] + geo_mesh.normals[-1, 1] * eta[j]

            # concatenate the 2 grids
            x = np.concat((x_wake1, x, x_wake2), axis=0)
            y = np.concat((y_wake1, y, y_wake2), axis=0)

            # get grid
            self.meshgrid = (x, y)


        # calculate velocity at each grid point
        # initialise with freestream velocity
        self.v = np.ones_like(self.meshgrid)
        self.v[0] *= config.V_inf_vec[0]
        self.v[1] *= config.V_inf_vec[1]

        # add in geometry influence
        for i in range(len(self.meshgrid[0][0, :])):
            for j in range(len(self.meshgrid[0][:, 0])):
                u1, v1 = flows.getConstantSourcePanelsVelocity(geo_mesh, (self.meshgrid[0][j, i], self.meshgrid[1][j, i]))
                u2, v2 = flows.getConstantVortexPanelsVelocity(geo_mesh, (self.meshgrid[0][j, i], self.meshgrid[1][j, i]))
                self.v[0][j, i] += u1 + u2
                self.v[1][j, i] += v1 + v2

    # ================ Velocity Plotting ================== #

    def plotStreamlines(self, colour='C0', density=1):

        if self.area_type == 'cartesian':
            if colour == 'velocity': # colour by velocity
                colour = np.sqrt(self.v[0]**2 + self.v[1]**2)
            
            # plot streamlines
            strm = plt.streamplot(self.meshgrid[0], self.meshgrid[1], self.v[0], self.v[1], color=colour, linewidth=1, density=density, cmap='inferno')
            plt.colorbar(strm.lines)

        elif self.area_type == 'conformal':
            x_vec = np.linspace(self.meshgrid[0].min(), self.meshgrid[0].max(), 100)
            y_vec = np.linspace(self.meshgrid[1].min(), self.meshgrid[1].max(), 100)
            x_cart, y_cart = np.meshgrid(x_vec, y_vec)
            
            # flatten
            points = np.vstack((self.meshgrid[0].flatten(), self.meshgrid[1].flatten())).T
            u_flat = self.v[0].flatten()
            v_flat = self.v[1].flatten()
            
            # interpolate
            u_cart = griddata(points, u_flat, (x_cart, y_cart), method='linear')
            v_cart = griddata(points, v_flat, (x_cart, y_cart), method='linear')

            if colour == 'velocity': # colour by velocity
                colour = np.sqrt(u_cart**2 + v_cart**2)

            # plot streamlines
            strm = plt.streamplot(x_cart, y_cart, u_cart, v_cart, color=colour, linewidth=1, density=density, cmap='inferno')
            plt.colorbar(strm.lines)

        # remove arrows if needed
        arrows = False
        if not arrows:
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
            contour = plt.contourf(self.meshgrid[0], self.meshgrid[1], 1 - (np.sqrt(self.v[0]**2 + self.v[1]**2)/self.config.V_inf)**2, levels=500, cmap='viridis')
        else:
            print("\nWarning!: No valid variable selected for contour plot.\n")
            return
        
        if colorbar:
            plt.colorbar(contour, label=var)
    
    def plotVelocityGrid(self):
        for i in range(self.meshgrid[0].shape[0]):
            plt.plot(self.meshgrid[0][i, :], self.meshgrid[1][i, :], color='gray', linewidth=0.5)
        for j in range(self.meshgrid[0].shape[1]):
            plt.plot(self.meshgrid[0][:, j], self.meshgrid[1][:, j], color='gray', linewidth=0.5)
    
class Visualisation:
    def __init__(self, config, mesh):
        
        #self.velocity_field = VelocityField(mesh, config, 'cartesian', (-0.02, 0.14), (-0.07, 0.07), 100, 100) # (0.095, 0.105), (-0.005, 0.005)
        self.velocity_field = VelocityField(mesh, config, 'conformal', ylim=0.05, ny=50)
        self.config = config
        self.mesh = mesh

    def plot(self):
        plt.figure()
        self.velocity_field.plotContour('pressure')
        #self.velocity_field.plotVelocityGrid()
        self.velocity_field.plotStreamlines(colour='velocity', density=2)
        self.mesh.plotGeometry(vertices=False, control_points=False, normals=False, tangents=False, boundary_layer=True, gcs=True, vectors_percent_scale=1)
        plt.axis('scaled')
        #plt.show()

        #plt.figure()
        #plt.plot(self.mesh.cp.flatten())
        plt.show()