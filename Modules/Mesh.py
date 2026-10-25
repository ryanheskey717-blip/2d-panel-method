r"""
Mesh Generation and Solver Module
=================================

Handles the discretization of airfoil geometry, generation of aerodynamic 
influence coefficient matrices, and execution of the coupled inviscid-viscous 
iterative solver.
"""
import csv
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import time
from copy import deepcopy

import Modules.ElementaryFlows as flows
import Modules.General as General
import Modules.Viscous as Viscous

class Mesh:
    r"""Creates and manages the 2D panel mesh and aerodynamic solver routines.

    This class parses the airfoil geometry, calculates panel normal and tangent 
    vectors, builds the aerodynamic influence coefficient (AIC) matrix, and 
    orchestrates the iterative inviscid and viscous solving process[cite: 8].

    Attributes:
        config (Config): Configuration object containing solver settings.
        vertices (ndarray): `(x, y)` coordinates of the panel endpoints.
        control_points (ndarray): `(x, y)` coordinates of panel centers offset by a small distance.
        normals (ndarray): Unit normal vectors for each panel.
        tangents (ndarray): Unit tangent vectors for each panel.
        lengths (ndarray): Scalar lengths of each panel.
        A (ndarray): The AIC matrix for the linear system.
        b (ndarray): The right-hand side boundary condition vector.
        transpiration_velocity (ndarray): Blowing velocities simulating boundary layer displacement.
        source_strengths (ndarray): Computed source strengths for each panel.
        vortex_strength (float): Computed uniform vortex strength satisfying the Kutta condition.
    """
    def __init__(self, config, A=None):
        r"""Initializes the Mesh object with the provided configuration.

        Args:
            config (Config): Configuration settings instance.
            A (ndarray, optional): Predefined A matrix (if provided, A is not calculated)
        """
        #import config
        self.config = config
        self.A = A

    def initialiseSetup(self, first=False):
        r"""Loads geometry, calculates panel properties, and preallocates arrays.

        Scales the input vertices by the chord length, calculates normalized 
        tangent and normal vectors, determines control point locations, and 
        initializes the $A$ matrix and $b$ vector.

        Args:
            first (bool, optional): Flag indicating if this is the first setup pass.
        """
        # ================ Start Everything ================ #
        # record start time and initialise
        self.start_time = time.time()
        self.last_time_check = 0
        self.converged = False
        self.iteration = 1

        # load geometry
        self.vertices = np.loadtxt(self.config.geo_file, skiprows=1)
        self.vertices *= self.config.chord

        # ============= Calculate control points, normals, tangents and lengths ============== #
        self.normals = []
        for i in range(len(self.vertices) - 1):
            self.normals.append([
                (self.vertices[i+1, 1] - self.vertices[i, 1]),
                -(self.vertices[i+1, 0] - self.vertices[i, 0]),
            ])
        self.normals = np.array(self.normals)
        self.normals /= np.linalg.norm(self.normals, axis=1, keepdims=True) # normalise

        self.tangents = []
        for i in range(len(self.vertices) - 1):
            self.tangents.append([
                -(self.vertices[i+1, 0] - self.vertices[i, 0]),
                -(self.vertices[i+1, 1] - self.vertices[i, 1]),
            ])
        self.tangents = np.array(self.tangents)
        self.lengths = np.linalg.norm(self.tangents, axis=1)
        self.tangents /= np.linalg.norm(self.tangents, axis=1, keepdims=True) # normalise

        self.control_points = []
        for i in range(len(self.vertices) - 1):
            self.control_points.append([
                (self.vertices[i, 0] + self.vertices[i+1, 0]) / 2, 
                (self.vertices[i, 1] + self.vertices[i+1, 1]) / 2
            ])
        self.control_points = np.array(self.control_points)
        self.control_points += self.normals * self.config.control_point_offset # offset a bit from surface

        # ================= Preallocate Arrays ==================== #
        self.N = len(self.control_points)
        if self.A is None: # check if A is provided for sweeps
            self.A_provided = False
            self.A = np.zeros((self.N+1, self.N+1)) # +1 for kutta condition
        else: # dont preallocate if it exists
            self.A_provided = True
        self.b = np.zeros(self.N+1)
        self.delta_star = np.zeros(self.N) # BL computed at each control point
        self.updateWakePoints(first=True)
        self.transpiration_velocity = np.zeros(self.N)
        self.normals_at_vertices = self.interpControlPointsToVertices(self.normals)
        self.normals_at_vertices /= np.linalg.norm(self.normals_at_vertices, axis=1, keepdims=True)
        
    # =================== Preconditioning =================== #
    def computeAandb(self):
        r"""Constructs the linear system for the panel method.

        Populates the aerodynamic influence coefficient matrix $A$ and boundary 
        condition vector $b$. Implements the zero-penetration (or transpiration) 
        boundary condition across all control points and applies the Kutta 
        condition at the trailing edge.
        """
        if not self.A_provided: # calculate A if it was not provided
            # section of A relating to no-penetration boundary condition
            for i in range(self.N): # control point
                for j in range(self.N + 1): # panel + influence from constant strength vortex sheet
                    if j == self.N: # vortex influence
                        u, v = flows.getConstantVortexPanelsVelocity(self, self.control_points[i], strength=1.0)
                        self.A[i, j] = np.dot(self.normals[i], np.array([u, v]))
                    else:
                        if i == j: # source self-influence
                            self.A[i, j] = 0.5
                        else:
                            u, v = flows.getConstantSourcePanelVelocity(self.vertices[j], self.vertices[j+1], self.control_points[i], strength=1.0)
                            self.A[i, j] = np.dot(self.normals[i], np.array([u, v]))

            # section of A relating to kutta condition (these find tangential influences)
            for j in range(self.N + 1): # panel + influence from constant strength vortex sheet
                if j == self.N: # vortex influence
                    u1, v1 = flows.getConstantVortexPanelsVelocity(self, self.control_points[0], strength=1.0)  # at upper TE control point
                    u2, v2 = flows.getConstantVortexPanelsVelocity(self, self.control_points[-1], strength=1.0) # at lower TE control point
                    self.A[-1, j] = (np.dot(self.tangents[0], np.array([u1, v1]))) + (np.dot(self.tangents[-1], np.array([u2, v2])))
                else:
                    u1, v1 = flows.getConstantSourcePanelVelocity(self.vertices[j], self.vertices[j+1], self.control_points[0], strength=1.0)  # at upper TE control point
                    u2, v2 = flows.getConstantSourcePanelVelocity(self.vertices[j], self.vertices[j+1], self.control_points[-1], strength=1.0) # at lower TE control point
                    self.A[-1, j] = (np.dot(self.tangents[0], np.array([u1, v1]))) + (np.dot(self.tangents[-1], np.array([u2, v2])))

        # RHS
        for i in range(self.N + 1): # control points + kutta condition
            if i == self.N: # kutta row
                self.b[-1] = - (np.dot(self.config.V_inf_vec, self.tangents[0]) + np.dot(self.config.V_inf_vec, self.tangents[-1])) # - (upper + lower)
            else:
                self.b[i] = - np.dot(self.config.V_inf_vec, self.normals[i]) + self.transpiration_velocity[i]


    # ================== Inviscid Solving ===================== #
    def solveInviscid(self):
        r"""Solves the linear system to find source strengths and vortex strength."""
        x = np.linalg.solve(self.A, self.b)
        self.source_strengths = x[:-1]
        self.vortex_strength = x[-1]
    
    # =================== Viscous Solving ===================== #
    def solveViscous(self, initial=False):
        r"""Calculates tangential velocities and updates viscous boundary layer properties.

        Computes the displacement thickness and corresponding transpiration velocity 
        for the next inviscid iteration if viscous coupling is enabled.

        Args:
            initial (bool, optional): Flag indicating if this is the initial viscous pass.
        """
        # calculate tangential velocity at control points and vertices
        self.calculateTangentialVelocityAtControlPoints() # TODO: maybe do this elsewhere do this func. does not need to be run when no viscous
        self.velocity_tangent_at_vertices = self.interpControlPointsToVertices(self.velocity_tangent_at_control_points)

        # call viscous solving functions
        if self.config.include_viscous:
            Viscous.updateDisplacementThickness(self)
            Viscous.updateTranspirationVelocity(self)
            self.skin_friction_coeff = Viscous.getSkinFrictionCoefficient(self)
        else:
            self.skin_friction_coeff = np.zeros(self.N)
    
    # ======================= General ========================= #
    def interpControlPointsToVertices(self, cp_vals):
        r"""Interpolates values from panel control points to panel vertices.

        Args:
            cp_vals (ndarray): Array of scalar values evaluated at control points.

        Returns:
            ndarray: Array of values interpolated to the vertices using inverse distance weighting.
        """
        # find distances of vertices to control points
        outs = []
        dist1 = 0; dist2 = 0; v1 = 0; v2 = 0
        for i in range(len(cp_vals)+1):
            if i == 0: # first vertex
                dist1 = self.lengths[-1] / 2
                dist2 = self.lengths[i] / 2
                v1 = cp_vals[-1]
                v2 = cp_vals[i]
            elif i == len(cp_vals): # last vertex
                dist1 = self.lengths[i-1] / 2
                dist2 = self.lengths[0] / 2
                v1 = cp_vals[i-1]
                v2 = cp_vals[0]
            else:
                dist1 = self.lengths[i-1] / 2
                dist2 = self.lengths[i] / 2
                v1 = cp_vals[i-1]
                v2 = cp_vals[i]
            w1 = 1 / dist1
            w2 = 1 / dist2
            outs.append((w1 * v1 + w2 * v2) / (w1 + w2))
        outs = np.array(outs)
        return outs

    def updateWakePoints(self, first=False): # TODO: only used for vis now (simplify)
        r"""Initializes or updates the spatial coordinates of the trailing edge wake. Only used for 
        visualisation purposes for now.

        Args:
            first (bool, optional): If True, generates the initial straight wake trajectory.
        """
        if first == True:
            start_point = self.vertices[self.config.wake_starting_index]
            self.wake_vertices = np.linspace(start_point, start_point + (self.config.wake_length, 0), self.config.wake_points) # TODO: is this better?: * self.config.V_inf_vec / self.config.V_inf
        else:
            pass #### TODO: calculate this based of last iters fluid flow
                 #### if this is done, need to update the grid calc in General.VelocityField.__init__()


    # ================= Flow Properties =================== #

    def calculateTangentialVelocityAtControlPoints(self):
        r"""Evaluates the total tangential velocity at each control point."""
        self.velocity_tangent_at_control_points = np.zeros(self.N)
        for i in range(self.N): # control point
            total_vel = self.config.V_inf_vec + flows.getConstantSourcePanelsVelocity(self, self.control_points[i]) + flows.getConstantVortexPanelsVelocity(self, self.control_points[i])
            self.velocity_tangent_at_control_points[i] = np.dot(total_vel, self.tangents[i])

    def calculatePressureOnPanels(self):
        r"""Delegates pressure calculation based on the configured method."""
        if self.config.pressure_calculation == "inviscid_bernoulli":
            # tangential velocity calculated in viscous solve
            self.pressureFromBernoulli()
        else:
            print(f"\nWarning: {self.pressure_calculation} not yet supported. Switching to Bernoulli.\n")
            # tangential velocity calculated in viscous solve
            self.pressureFromBernoulli()

    def pressureFromBernoulli(self): # dimensional
        r"""Calculates dimensional pressure and the pressure coefficient $C_p$ using Bernoulli's equation."""
        self.pressure = self.config.p_inf - 0.5 * self.config.rho_inf * ( self.velocity_tangent_at_control_points ** 2 - self.config.V_inf ** 2 )
        self.cp = 1 - (self.velocity_tangent_at_control_points / self.config.V_inf) ** 2

    def calculateForcesAndMoments(self):
        r"""Calculates the total lift, drag, and moment coefficients from pressure and viscous forces."""
        self.total_force = self.pressureForce() + self.viscousForce()
        self.total_moment = self.pressureMoment() + self.viscousMoment()

        self.c_force = self.total_force / ( 0.5 * self.config.rho_inf * self.config.V_inf ** 2 * self.config.chord)
        self.c_moment = self.total_moment / ( 0.5 * self.config.rho_inf * self.config.V_inf ** 2 * self.config.chord ** 2)

        self.lift = np.dot(self.total_force, np.array([-np.sin(self.config.alpha),np.cos(self.config.alpha)]))
        self.drag = np.dot(self.total_force, np.array([np.cos(self.config.alpha), np.sin(self.config.alpha)]))
        self.c_lift = np.dot(self.c_force, np.array([-np.sin(self.config.alpha), np.cos(self.config.alpha)]))
        self.c_drag = np.dot(self.c_force, np.array([np.cos(self.config.alpha), np.sin(self.config.alpha)]))

    def pressureForce(self): # sum integrate for forces (as only know pressure at control points i.e. centres)
        r"""Integrates the pressure distribution to find the net inviscid force vector.

        Returns:
            ndarray: The $(F_x, F_y)$ force components generated by panel pressures.
        """
        fx = np.sum( - self.pressure * self.lengths * self.normals[:, 0] )
        fy = np.sum( - self.pressure * self.lengths * self.normals[:, 1] )

        return np.array([fx, fy])

    def viscousForce(self): #### TODO: investigate issue of discontinuous force vs. aoa (i think due to i_stag changing)
        r"""Integrates the skin friction distribution to find the net viscous force vector.

        Returns:
            ndarray: The $(F_x, F_y)$ force components generated by skin friction.
        """
        if self.config.include_viscous:
            q = 0.5 * self.config.rho_inf * self.config.V_inf**2
            fx = 0; fy = 0
            for i in np.arange(self.i_stag, -1, -1): # top surface
                force = q * self.lengths[i] * self.skin_friction_coeff[i]
                fx += force * self.tangents[i, 0]
                fy += force * self.tangents[i, 1]
            for i in range(self.i_stag+1, self.N): # bottom surface
                force = q * self.lengths[i] * self.skin_friction_coeff[i]
                fx += force * -self.tangents[i, 0]
                fy += force * -self.tangents[i, 1]

            return np.array([fx, fy])
        else:
            return np.array([0, 0])

    def pressureMoment(self):
        r"""Calculates the pitching moment generated by pressure forces around the reference point.

        Returns:
            float: The net aerodynamic moment from inviscid pressure.
        """
        r = self.control_points - self.config.ref_position
        forces = - self.pressure[:, None] * self.normals * self.lengths[:, None]

        return np.sum(r[:, 0] * forces[:, 1] - r[:, 1] * forces[:, 0])

    def viscousMoment(self):
        r"""Calculates the pitching moment generated by viscous forces around the reference point.

        Returns:
            float: The net aerodynamic moment from skin friction.
        """
        if self.config.include_viscous:
            r = self.control_points - self.config.ref_position
            
            q = 0.5 * self.config.rho_inf * self.config.V_inf**2
            forces = np.zeros((self.N, 2))
            for i in np.arange(self.i_stag, -1, -1): # top surface
                force = q * self.lengths[i] * self.skin_friction_coeff[i]
                forces[i, 0] += force * self.tangents[i, 0]
                forces[i, 1] = force * self.tangents[i, 1]
            for i in range(self.i_stag+1, self.N): # bottom surface
                force = q * self.lengths[i] * self.skin_friction_coeff[i]
                forces[i, 0] = force * -self.tangents[i, 0]
                forces[i, 1] = force * -self.tangents[i, 1]

            return np.sum(r[:, 0] * forces[:, 1] - r[:, 1] * forces[:, 0])
        else:
            return 0

    # ====================== Running =========================== #
    def run(self):
        r"""Executes a single iteration of the solver.

        Preconditions the matrix, solves the inviscid flow, updates viscous models, 
        evaluates pressure forces, and checks convergence.
        """
        if self.config.verbose:
            self.printNewCase()

        if self.config.verbose:
            self.printAfterSetup()

        # Setup and Solve inviscid
        self.computeAandb()
        if self.config.verbose:
            self.printAfterPopulation()

        # Solve linear system
        self.solveInviscid()
        if self.config.verbose:
            self.printAfterInviscidSolve()
        
        # Solve viscous
        self.solveViscous()
        if self.config.verbose and self.config.include_viscous:
            self.printAfterViscousSolve()
        if self.config.verbose and not self.config.include_viscous:
            self.printAfterNoViscous()

        #### TODO: alter anything based on viscous solve????

        # Solution Analysis
        self.calculatePressureOnPanels()
        self.calculateForcesAndMoments()
        self.checkConvergence()
        if self.config.verbose:
            self.printForcesResults()

        #### TODO: write convergence to file here?
        self.updateResults()

    def run_case(self):
        r"""Main execution loop that drives the solver to convergence.

        Iteratively calls `run()` until the convergence criteria are met, then 
        handles output and visualization generation.
        """
        # calculate mesh parameters
        self.initialiseSetup()
        if self.config.verbose:
            self.printInitial()

        # main iteration loop
        while not self.converged:
            self.run()
            self.iteration += 1

        # Output
        if self.config.write_to_file:
            if self.config.verbose:
                self.printBeforeWriting()
            if self.config.sweep_iteration != 0:
                self.resultsToCSV(first_entry=False)
            else:
                self.resultsToCSV()
            if self.config.verbose:
                self.printAfterWriting()
        
        # Visualisation
        if self.config.visualisation:
            if self.config.verbose:
                self.printBeforeVelField()
            self.visualisation = General.Visualisation(self.config, self)
            if self.config.verbose:
                self.printAfterVelField()
            self.visualisation.plot()
            ###### TODO: auto size arrows
        
        # Exit
        if self.config.verbose:
            self.printFinal()

    def checkConvergence(self):
        r"""Evaluates iteration counts and solver state to flag completion."""
        if self.iteration == 2 and not self.config.run_till_converged and self.config.include_viscous:
            self.converged = True
        elif self.iteration == 1 and not self.config.run_till_converged and not self.config.include_viscous:
            self.converged = True
        else:
            pass
            #raise('more than 1 iter not supported yet')
    
    # ===================== Output ========================= #

    def updateResults(self):
        r"""Aggregates flow variables and integrated forces into a results list."""
        self.results = [self.config.geo_file, self.config.M_inf, self.config.V_inf, self.config.Re, np.rad2deg(self.config.alpha), self.config.chord, self.config.ref_position[0], self.config.ref_position[1],
                        self.total_force[0], self.total_force[1], self.c_force[0], self.c_force[1], self.drag, self.lift, self.c_drag, self.c_lift, self.total_moment, self.c_moment]

    def resultsToCSV(self, first_entry=True):
        r"""Writes the current aerodynamic results to a CSV file.

        Args:
            first_entry (bool, optional): If True, overwrites the file and writes headers. 
                If False, appends a new row of data.
        """
        # set writing type
        if first_entry == False:
            mode = 'a' # append (next case saves below previous)
        else:
            mode = 'w' # overwrite (first entry)
        
        # open and write to file
        with open(self.config.out_coeff_file, mode, newline='') as file:
            writer = csv.writer(file)

            # header
            if first_entry:
                writer.writerow(['geometry', 'mach', 'v_inf', 're', 'alpha_deg', 'chord', 'ref_loc_x', 'ref_loc_y', 'f_x', 'f_y', 'c_x', 'c_y', 'f_d', 'f_l', 'c_d', 'c_l', 'moment', 'c_m'])

            # write data
            writer.writerow(self.results)

    # =================== Printing ====================== #
    def printInitial(self):
        r"""Prints an update when verbose is set to true."""
        print( '')
        print( '~' * 60)
        print( '2D Panel Method for Airfoils'.center(60))
        print( '    Ryan Heskey'.center(60))
        print( '~' * 60)
        print( '')
        print( 'Input:')
        print(f'    Recieved Geometry File: {self.config.geo_file} ({len(self.vertices)} Vertices)')
        print(f'    Non-Dimensional Numbers: Re = {self.config.Re:.0f}, M = {self.config.M_inf}')
        print(f'    Pressure Calculation Mode: {self.config.pressure_calculation}')
        print( '')
    
    def printNewCase(self):
        r"""Prints an update when verbose is set to true."""
        print( '~' * 22, f' Iteration {self.iteration:.0f} ', '~' * 23)
        print( '')
        print( 'Setting up viscous correction ...', end='', flush=True)
        self.last_time_check = time.time()
    
    def printAfterSetup(self):
        r"""Prints an update when verbose is set to true."""
        print(f' Done ({time.time() - self.last_time_check:.2f} s)')
        print( '')
        print( 'Populating A and b ...', end='', flush=True)
        self.last_time_check = time.time()

    def printAfterPopulation(self):
        r"""Prints an update when verbose is set to true."""
        print(f' Done ({time.time() - self.last_time_check:.2f} s)')
        print( '')
        print( 'Solving inviscid ...', end='', flush=True)
        self.last_time_check = time.time()

    def printAfterInviscidSolve(self):
        r"""Prints an update when verbose is set to true."""
        print(f' Done ({time.time() - self.last_time_check:.2f} s)')
        print( '')
        print( 'Solving viscous (if applicable) ...', end='', flush=True)
        self.last_time_check = time.time()

    def printAfterViscousSolve(self):
        r"""Prints an update when verbose is set to true."""
        print(f' Done ({time.time() - self.last_time_check:.2f} s)')
        print( '')
        print( 'Calculating forces ...', end='', flush=True)
        self.last_time_check = time.time()

    def printAfterNoViscous(self):
        r"""Prints an update when verbose is set to true."""
        print(f' Skipped ({time.time() - self.last_time_check:.2f} s)')
        print( '')
        print( 'Calculating forces ...', end='', flush=True)
        self.last_time_check = time.time()
    
    def printForcesResults(self):
        r"""Prints an update when verbose is set to true."""
        print(f' Done ({time.time() - self.last_time_check:.2f} s)')
        print( '')
        print(f'Results Summary: ({"Converged" if self.converged else "Not Converged"})')
        print( '')
        print(f'    Cx: {self.c_force[0]:.3f} ({self.total_force[0]:.2f} N/m)')
        print(f'    Cy: {self.c_force[1]:.3f} ({self.total_force[1]:.2f} N/m)')
        print( '')
        print(f'    Cd: {self.c_drag:.3f} ({self.drag:.2f} N/m)')
        print(f'    Cl: {self.c_lift:.3f} ({self.lift:.2f} N/m)')
        print( '')
        print(f'    Cm: {self.c_moment:.3f} ({self.total_moment:.2f} N.m/m) around (x, y) = ({self.config.ref_position[0]}, {self.config.ref_position[1]}) m')
        print( '')
        print( '~' * 60)
        print( '')

    def printBeforeWriting(self):
        r"""Prints an update when verbose is set to true."""
        print( 'Writing results to file ...', end='', flush=True)
        self.last_time_check = time.time()
    
    def printAfterWriting(self):
        r"""Prints an update when verbose is set to true."""
        print(f' Done ({time.time() - self.last_time_check:.2f} s)')
        print( '')
    
    def printBeforeVelField(self):
        r"""Prints an update when verbose is set to true."""
        print( 'Creating visualisation ...', end='', flush=True)
        self.last_time_check = time.time()
    
    def printAfterVelField(self):
        r"""Prints an update when verbose is set to true."""
        print(f' Done ({time.time() - self.last_time_check:.2f} s)')
        print( '')
    
    def printFinal(self):
        r"""Prints an update when verbose is set to true."""
        print( 'Exiting ... Done')
        print( '')
        print(f'Total Runtime: {time.time() - self.start_time:.3f} s')
        print( '')
        print( '~' * 60)
        print( '')