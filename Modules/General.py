"""
General
================

This module provides the classes for general setup and visualisation. This module is used
for setting up the necessary parts to run and analyse the results of the panel method.
"""
import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

import Modules.ElementaryFlows as flows

# ==================== Loading =================== #

# load and store config
class Config:
    r"""Loads, stores, and processes configuration parameters for the panel method solver.

    This class reads input settings from a specified YAML configuration file,
    initializes flow conditions and solver parameters, and computes derived
    aerodynamic properties.

    Attributes:
        full_data (dict): The complete raw data dictionary loaded from the YAML file.
        verbose (bool): Flag to enable detailed print outputs.
        geo_file (str): Path to the airfoil geometry file.
        chord (float): Chord length of the airfoil.
        ref_position (ndarray): `(x, y)` coordinate array for the aerodynamic reference position.
        M_inf (float): Freestream Mach number.
        T_inf (float): Freestream temperature in Celsius.
        rho_inf (float): Freestream density $\rho$.
        mu_inf (float): Freestream dynamic viscosity $\mu$.
        p_inf (float): Freestream static pressure.
        alpha (float): Angle of attack $\alpha$ in radians (converted internally from degrees).
        control_point_offset (float): Offset distance for panel collocation points.
        convergence_criterion (float): Tolerance threshold for solver convergence.
        include_viscous (bool): Flag indicating if viscous boundary layer coupling is active.
        boundary_layer_calculation (str): Type of boundary layer model to use (e.g., 'none', 'blasius').
        inflation_layers (int): Number of inflation layers for the mesh or wake.
        wake_starting_index (int): Panel index where the downstream wake begins.
        wake_points (int): Number of discrete points defining the wake downstream.
        wake_length (float): Total length of the simulated wake.
        pressure_calculation (str): Method used for computing pressure coefficients.
        visualisation (bool): Flag to enable plot generation and geometry visualization.
        scale (float): Scaling factor for geometry visualization.
        write_to_file (bool): Flag to save output data to a file.
        overwrite_output (bool): Flag to allow overwriting existing output files.
        out_coeff_file (str): File path for saving aerodynamic coefficients.
        sweep_iteration (int): Counter for the current alpha or parameter sweep iteration.
        nu_inf (float): Freestream kinematic viscosity $\nu$.
        V_inf (float): Freestream velocity magnitude.
        V_inf_vec (ndarray): `(u, v)` freestream velocity vector components.
        Re (float): Reynolds number based on the chord length.
        run_till_converged (bool): Flag determining if the solver loops until convergence or runs a single iteration.
    """
    def __init__(self, input_file):
        r"""Initializes the Config object by parsing a YAML configuration file.

        Args:
            input_file (str): Path to the YAML configuration file containing the solver settings.
        """
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
        self.p_inf = self.full_data["p_inf"]
        self.alpha = self.full_data["alpha"]

        # solver
        self.control_point_offset = float(self.full_data["control_point_offset"])
        self.convergence_criterion = float(self.full_data["convergence_criterion"])
        
        # turbulence/viscous
        self.include_viscous = True
        self.boundary_layer_calculation = self.full_data["boundary_layer"]
        self.inflation_layers = self.full_data["inflation_layers"]
        self.wake_starting_index = int(self.full_data["wake_starting_index"])
        self.wake_points = int(self.full_data["wake_points"])
        self.wake_length = float(self.full_data["wake_length"])

        # flow properties
        self.pressure_calculation = self.full_data["pressure_calculation"]

        # visualisation
        self.visualisation = self.full_data["visualisation"]
        self.vis_grid_method = self.full_data["grid_method"]
        self.vis_geo = self.full_data["vis_geo"]
        self.strm_colour = self.full_data["streamlines_colour"]
        self.cont_type = self.full_data["contour_type"]
        self.scale = self.full_data["scale"]

        # output
        self.write_to_file = self.full_data["output"]["write_to_file"]
        self.overwrite_output = self.full_data["output"]["overwrite"]
        self.out_coeff_file = self.full_data["output"]["coefficients"]

        # set sweep run number (iterated in Sweep class in PanelMethod)
        self.sweep_iteration = 0

        # calculate other important values
        self.update()

    def update(self):
        r"""Computes derived aerodynamic parameters and updates solver states.

        Calculates kinematic viscosity ($\nu$), converts the input angle of attack 
        ($\alpha$) to radians, computes the freestream velocity vector using ideal 
        gas relations, and calculates the Reynolds number ($Re$). It also evaluates 
        configuration flags to determine if viscous boundary layer calculations and 
        iterative convergence loops should be enabled.
        """
        # calculated values
        self.nu_inf = self.mu_inf / self.rho_inf
        self.alpha = np.deg2rad(self.alpha)
        self.V_inf = self.M_inf * np.sqrt(1.4 * 287 * (self.T_inf + 273.15))
        self.V_inf_vec = np.array([
            self.V_inf * np.cos(self.alpha),
            self.V_inf * np.sin(self.alpha)
        ])
        self.Re = self.V_inf * self.chord * self.rho_inf / self.mu_inf

        if self.boundary_layer_calculation == 'none': 
            self.include_viscous = False # defaults to True if this^ is not true

        if self.convergence_criterion == 0.0: # single iteration
            self.run_till_converged = False
        else: # let run until convergence
            self.run_till_converged = True


# ==================== Analysing ========================= #

class VelocityField:
    r"""Generates a grid of off-body points, and calculates the velocity at each of these to visualise
    the flow around the solved body.

    """
    def __init__(self, geo_mesh, config, area_type, xlim=None, ylim=None, nx=None, ny=None):
        r"""Initializes the VelocityField object by creating a grid around the aerodynamic body
        and calculates the velocity at each point.

        Args:
            geo_mesh (object): Instance of Mesh class in which the panels are defined.
            config (object): Instance of the Config class containing settings.
            aera_type (str): 'cartesian' x, y rectangle OR 'conformal' wraps around body.
            xlim (tuple): domain of cartesian grid.
            ylim (tuple): range of cartesian grid.
            nx (int): number of x points.
            ny (int): number of y points.
        """
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
        r"""Plots streamlines on a preexisting figure.

        Args:
            color (str): The colour of the lines ('velocity' to be coloured byt the velocity at each point)
            density (int, float): How dense should the streamlines be.
        """
        if self.area_type == 'cartesian':
            if colour == 'velocity': # colour by velocity
                colour = np.sqrt(self.v[0]**2 + self.v[1]**2)
            
            # plot streamlines
            strm = plt.streamplot(self.meshgrid[0], self.meshgrid[1], self.v[0], self.v[1], color=colour, linewidth=1, density=density, cmap='inferno')
            
            if colour == 'velocity': # add colour bar if needed
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
            
            if colour == 'velocity': # add colour bar if needed
                plt.colorbar(strm.lines)

        # remove arrows if needed
        arrows = False
        if not arrows:
            ax = plt.gca()
            for art in ax.get_children():
                if isinstance(art, plt.matplotlib.patches.FancyArrowPatch):
                    art.remove()

    def plotContour(self, var, colorbar=True):
        r"""Plots contours of various variables on a preexisting figure.

        Args:
            var (str): What variable should the contour be based on ('vx', 'vy', 'vmag', 'pressure')
            colorbar (bool): Include colorbar or not (defaults to True)
        """
        if var == "vx":
            contour = plt.contourf(self.meshgrid[0], self.meshgrid[1], self.v[0], levels=50, cmap='viridis')
        elif var == "vy":
            contour = plt.contourf(self.meshgrid[0], self.meshgrid[1], self.v[1], levels=50, cmap='viridis')
        elif var == "vmag":
            contour = plt.contourf(self.meshgrid[0], self.meshgrid[1], np.sqrt(self.v[0]**2 + self.v[1]**2), levels=50, cmap='viridis')
        elif var == "pressure":
            contour = plt.contourf(self.meshgrid[0], self.meshgrid[1], 1 - (np.sqrt(self.v[0]**2 + self.v[1]**2)/self.config.V_inf)**2, levels=500, cmap='viridis')
        else:
            print("\nWarning!: No valid variable selected for contour plot.\n")
            return
        
        if colorbar:
            plt.colorbar(contour, label=var)
    
    def plotVelocityGrid(self):
        r"""Plots the off-body grid points at which the velocity field has been calculated at 
        on a preexisting figure.
        """
        for i in range(self.meshgrid[0].shape[0]):
            plt.plot(self.meshgrid[0][i, :], self.meshgrid[1][i, :], color='gray', linewidth=0.5)
        for j in range(self.meshgrid[0].shape[1]):
            plt.plot(self.meshgrid[0][:, j], self.meshgrid[1][:, j], color='gray', linewidth=0.5)
    
class Visualisation:
    r"""Handles the generation of a visualisation of the flow around a body."""
    def __init__(self, config, mesh):
        r"""Initialises a visualisation and accepts the info required to create it. Creates a 
        VelocityField instance to calculate off-body velocities.

        Args:
            config (object): Instance of Config class, containing settings.
            mesh (object): Instance of Mesh class, containing geometry and solved panel system.
        """
        self.config = config
        self.mesh = mesh
        if self.config.vis_grid_method == 'cartesian':
            self.velocity_field = VelocityField(mesh, config, 'cartesian', (0.095, 0.105), (-0.005, 0.005), 100, 100) # (-0.02, 0.14), (-0.07, 0.07)
        elif self.config.vis_grid_method == 'conformal':
            self.velocity_field = VelocityField(mesh, config, 'conformal', ylim=0.2, ny=50)
        else:
            print(f"\nWarning: {self.config.vis_grid_method} not supported. Switching to conformal.\n")

    def plotGeometry(self, vertices=False, control_points=False, normals=False, tangents=False, boundary_layer=False, gcs=False, vectors_percent_scale=100):
        r"""Generates diagnostic plots of the panel mesh and boundary layer.

        Args:
            vertices (bool): Flag to scatter plot panel endpoints.
            control_points (bool): Flag to scatter plot control points.
            normals (bool): Flag to plot normal vectors via quiver.
            tangents (bool): Flag to plot tangent vectors via quiver.
            boundary_layer (bool): Flag to plot the displacement thickness overlay.
            gcs (bool): Flag to plot the global coordinate system axes.
            vectors_percent_scale (float): Scaling factor for vector visualization.
        """
        plt.plot(self.mesh.vertices[:, 0], self.mesh.vertices[:, 1], color='black', linewidth=1)
        plt.plot(self.mesh.wake_vertices[:, 0], self.mesh.wake_vertices[:, 1], color='black', linewidth=1)
        if vertices:
            plt.scatter(self.mesh.vertices[:, 0], self.mesh.vertices[:, 1], color='black', s=3)
        if control_points:
            plt.scatter(self.mesh.control_points[:, 0], self.mesh.control_points[:, 1], color='red', s=3)
        if normals:
            plt.quiver(self.mesh.control_points[:, 0], self.mesh.control_points[:, 1], self.mesh.normals[:, 0], self.mesh.normals[:, 1], angles='xy', scale_units='xy', scale=100/vectors_percent_scale, width=0.005, color='g')
        if tangents:
            plt.quiver(self.mesh.control_points[:, 0], self.mesh.control_points[:, 1], self.mesh.tangents[:, 0], self.mesh.tangents[:, 1], angles='xy', scale_units='xy', scale=100/vectors_percent_scale, width=0.005, color='b')
        if boundary_layer:
            plt.plot(self.mesh.control_points[:, 0] + self.mesh.delta_star * self.mesh.normals[:, 0], self.mesh.control_points[:, 1] + self.mesh.delta_star * self.mesh.normals[:, 1], color='red', linewidth=1)
        if gcs:
            self.plotGlobalCoordinateSystem(size=vectors_percent_scale)

    def plotGlobalCoordinateSystem(self, size=100):
        r"""Draws the global $X-Y$ coordinate origin.

        Args:
            size (float): Scaling parameter for axis length.
        """
        plt.quiver(0, 0, 1, 0, angles='xy', scale_units='xy', scale=100/size, width=0.005, color='r') # x
        plt.text(1.05 * size/100, 0, "x", color='r', ha='left', va='center', fontweight='bold')
        plt.quiver(0, 0, 0, 1, angles='xy', scale_units='xy', scale=100/size, width=0.005, color='b') # y
        plt.text(0, 1.05 * size/100, "y", color='b', ha='center', va='bottom', fontweight='bold')

    def plot(self):
        r"""Plots the visualisation of the body and calculated off body variables.
        """
        plt.figure()
        if self.config.cont_type != 'none':
            self.velocity_field.plotContour(self.config.cont_type)
        #self.velocity_field.plotVelocityGrid()
        if self.config.strm_colour != 'none':
            self.velocity_field.plotStreamlines(colour=self.config.strm_colour, density=1)
        self.plotGeometry(vertices=self.config.vis_geo[0], control_points=self.config.vis_geo[1], normals=self.config.vis_geo[2], tangents=self.config.vis_geo[3], boundary_layer=self.config.vis_geo[4], gcs=self.config.vis_geo[5], vectors_percent_scale=2)
        plt.axis('scaled')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.show()