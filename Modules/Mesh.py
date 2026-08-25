# class for creating and storing all mesh values, including wake

import csv
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import time
from copy import deepcopy

import Modules.ElementaryFlows as flows
import Modules.General as General

class Mesh:
    def __init__(self, config):
        # record start time and initialise
        self.start_time = time.time()
        self.last_time_check = 0
        self.converged = False
        self.iteration = 1

        # ============== Input Values ================ #
        self.config = config

        # ============== Load Geometry =============== #
        self.vertices = np.loadtxt(self.config.geo_file, skiprows=1)
        self.vertices *= self.config.chord

    def getMeshParameters(self, vertices, first=False):
        # ============= Calculate control points, normals, tangents and lengths ============== #
        self.normals = []
        for i in range(len(vertices) - 1):
            self.normals.append([
                (vertices[i+1, 1] - vertices[i, 1]),
                -(vertices[i+1, 0] - vertices[i, 0]),
            ])
        self.normals = np.array(self.normals)
        self.normals /= np.linalg.norm(self.normals, axis=1, keepdims=True) # normalise

        self.tangents = []
        for i in range(len(vertices) - 1):
            self.tangents.append([
                -(vertices[i+1, 0] - vertices[i, 0]),
                -(vertices[i+1, 1] - vertices[i, 1]),
            ])
        self.tangents = np.array(self.tangents)
        self.lengths = np.linalg.norm(self.tangents, axis=1)
        self.tangents /= np.linalg.norm(self.tangents, axis=1, keepdims=True) # normalise

        self.control_points = []
        for i in range(len(vertices) - 1):
            self.control_points.append([
                (vertices[i, 0] + vertices[i+1, 0]) / 2, 
                (vertices[i, 1] + vertices[i+1, 1]) / 2
            ])
        self.control_points = np.array(self.control_points)
        self.control_points += self.normals * self.config.control_point_offset # offset a bit from surface

        # ================= Preallocate Arrays ==================== #
        if first:
            self.N = len(self.control_points)
            self.A = np.zeros((self.N+1, self.N+1)) # +1 for kutta condition
            self.b = np.zeros(self.N+1)
            self.delta_star = np.zeros(self.N) # BL computed at each control point
            self.updateWakePoints(first)
    
    def saveGeometryInfo(self):
        self.getMeshParameters(self.vertices, first=True)
        self.geo_vertices = deepcopy(self.vertices)
        self.geo_normals = deepcopy(self.normals)
        self.geo_tangents = deepcopy(self.tangents)
        self.geo_lengths = deepcopy(self.lengths)
        self.geo_control_points = deepcopy(self.control_points)
        self.geo_normals_at_vertices = self.interpControlPointsToVertices(self.geo_normals)
        self.geo_normals_at_vertices /= np.linalg.norm(self.geo_normals_at_vertices, axis=1, keepdims=True)

    # =================== Preconditioning =================== #
    def computeAandb(self):

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
                self.b[i] = - np.dot(self.config.V_inf_vec, self.normals[i])


    # ================== Inviscid Solving ===================== #
    def solveInviscid(self):
        x = np.linalg.solve(self.A, self.b)
        self.source_strengths = x[:-1]
        self.vortex_strength = x[-1]
    
    # ================= Viscous Solving ==================== #
    def solveViscous(self):
        
        if self.iteration == 1: # use blasius solution
            pass
        else:
            pass

    def updateBLThickness(self):
        if self.iteration == 1: # blasis displacement thickness
            self.delta_star = 1.72 * self.geo_control_points[:, 0] / np.sqrt(self.config.V_inf * (self.geo_control_points[:, 0]+1e-10) / self.config.nu_inf) # +1e-10 for when x=0, make sure it returns zero
        else:
            pass #### TODO: add proper boundary layer solving here (how?)

        # interpolate
        self.delta_star_at_vertices = self.interpControlPointsToVertices(self.delta_star)
        
        # update vertices
        self.vertices = self.geo_vertices + self.delta_star_at_vertices[:, np.newaxis] * self.geo_normals_at_vertices
    
    def interpControlPointsToVertices(self, cp_vals):
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

    def updateWakePoints(self, first=False):
        if first == True:
            start_point = self.vertices[self.config.wake_starting_index]
            self.wake_vertices = np.linspace(start_point, start_point + (self.config.wake_length, 0), self.config.wake_points) # TODO: is this better?: * self.config.V_inf_vec / self.config.V_inf
        else:
            pass #### TODO: calculate this based of last iters fluid flow (ensure to use geo_vertices to start)


    # ================= Flow Properties =================== #

    def calculateTangentialVelocityAtControlPoints(self):
        self.velocity_tangent_at_control_points = np.zeros(self.N)
        for i in range(self.N): # control point
            total_vel = self.config.V_inf_vec + flows.getConstantSourcePanelsVelocity(self, self.control_points[i]) + flows.getConstantVortexPanelsVelocity(self, self.control_points[i])
            self.velocity_tangent_at_control_points[i] = np.dot(total_vel, self.tangents[i])

    def calculatePressureOnPanels(self):
        if self.config.pressure_calculation == "inviscid_bernoulli":
            self.calculateTangentialVelocityAtControlPoints()
            self.pressureFromBernoulli()
        else:
            print(f"\nWarning: {self.pressure_calculation} not yet supported.\n")

    def pressureFromBernoulli(self): # dimensional
        self.pressure = self.config.p_inf - 0.5 * self.config.rho_inf * ( self.velocity_tangent_at_control_points ** 2 - self.config.V_inf ** 2 )
        self.cp = 1 - (self.velocity_tangent_at_control_points / self.config.V_inf) ** 2

    def calculateForcesAndMoments(self):
        self.total_force = self.pressureForce() + self.viscousForce()
        self.total_moment = self.pressureMoment() + self.viscousMoment()

        self.c_force = self.total_force / ( 0.5 * self.config.rho_inf * self.config.V_inf ** 2 * self.config.chord)
        self.c_moment = self.total_moment / ( 0.5 * self.config.rho_inf * self.config.V_inf ** 2 * self.config.chord ** 2)

        self.lift = np.dot(self.total_force, np.array([-np.sin(self.config.alpha),np.cos(self.config.alpha)]))
        self.drag = np.dot(self.total_force, np.array([np.cos(self.config.alpha), np.sin(self.config.alpha)]))
        self.c_lift = np.dot(self.c_force, np.array([-np.sin(self.config.alpha), np.cos(self.config.alpha)]))
        self.c_drag = np.dot(self.c_force, np.array([np.cos(self.config.alpha), np.sin(self.config.alpha)]))

    def pressureForce(self): # midpoint integrate for forces (as only know pressure at control points i.e. centres)
        
        fx = np.sum( - self.pressure * self.lengths * self.geo_normals[:, 0] )
        fy = np.sum( - self.pressure * self.lengths * self.geo_normals[:, 1] )

        return np.array([fx, fy])

    def viscousForce(self):
        return np.array([0, 0])

    def pressureMoment(self):

        r = self.geo_control_points - self.config.ref_position
        forces = - self.pressure[:, None] * self.geo_normals * self.geo_lengths[:, None]

        return np.sum(r[:, 0] * forces[:, 1] - r[:, 1] * forces[:, 0])

    def viscousMoment(self):
        return 0

    # ====================== Running =========================== #
    def run(self):
        if self.config.verbose:
            self.printNewCase()

        # Solve viscous (blasius first iteration)
        self.solveViscous()
        self.updateBLThickness()
        if self.config.verbose:
            self.printAfterViscousSolve()

        # calculate mesh parameters
        self.getMeshParameters(vertices=self.vertices)
        if self.config.verbose:
            self.printAfterSetup()

        # Setup and Solve inviscid
        self.computeAandb()
        if self.config.verbose:
            self.printAfterPopulation()

        self.solveInviscid()
        if self.config.verbose:
            self.printAfterInviscidSolve()

        # Solution Analysis
        self.calculatePressureOnPanels()
        self.calculateForcesAndMoments()
        self.checkConvergence()
        if self.config.verbose:
            self.printForcesResults()

        #### TODO: write convergence to file here?

    def run_case(self):

        # Initial setup
        self.saveGeometryInfo()
        if self.config.verbose:
            self.printInitial()

        # main iteration loop
        while not self.converged:
            self.run()

        # Output
        if self.config.write_to_file:
            if self.config.verbose:
                self.printBeforeWriting()
            ##### add writing here
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
        
        if not self.config.run_till_converged and self.iteration == 1:
            self.converged = True
        else:
            raise('success')


    # ==================== Plotting ======================== #

    def plotGeometry(self, vertices=False, control_points=False, normals=False, tangents=False, boundary_layer=False, gcs=False, vectors_percent_scale=100):
        plt.plot(self.geo_vertices[:, 0], self.geo_vertices[:, 1], color='black', linewidth=1)
        plt.plot(self.wake_vertices[:, 0], self.wake_vertices[:, 1], color='black', linewidth=1)
        if vertices:
            plt.scatter(self.geo_vertices[:, 0], self.geo_vertices[:, 1], color='black', s=3)
        if control_points:
            plt.scatter(self.geo_control_points[:, 0], self.geo_control_points[:, 1], color='red', s=3)
        if normals:
            plt.quiver(self.geo_control_points[:, 0], self.geo_control_points[:, 1], self.geo_normals[:, 0], self.geo_normals[:, 1], angles='xy', scale_units='xy', scale=100/size, width=0.005, color='g')
        if tangents:
            plt.quiver(self.geo_control_points[:, 0], self.geo_control_points[:, 1], self.geo_tangents[:, 0], self.geo_tangents[:, 1], angles='xy', scale_units='xy', scale=100/size, width=0.005, color='b')
        if boundary_layer:
            plt.plot(self.geo_control_points[:, 0] + self.delta_star * self.geo_normals[:, 0], self.geo_control_points[:, 1] + self.delta_star * self.geo_normals[:, 1], color='red', linewidth=1)
        if gcs:
            self.plotGlobalCoordinateSystem(size=vectors_percent_scale)

    def plotGlobalCoordinateSystem(self, size=100):
        plt.quiver(0, 0, 1, 0, angles='xy', scale_units='xy', scale=100/size, width=0.005, color='r') # x
        plt.text(1.05 * size/100, 0, "x", color='r', ha='left', va='center', fontweight='bold')
        plt.quiver(0, 0, 0, 1, angles='xy', scale_units='xy', scale=100/size, width=0.005, color='b') # y
        plt.text(0, 1.05 * size/100, "y", color='b', ha='center', va='bottom', fontweight='bold')
    

    # =================== Printing ====================== #
    def printInitial(self):
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
        print( '~' * 22, f' Iteration {self.iteration:.0f} ', '~' * 23)
        print( '')
        print( 'Solving Viscous ...', end='', flush=True)
        self.last_time_check = time.time()

    def printAfterViscousSolve(self):
        print(f' Done ({time.time() - self.last_time_check:.2f} s)')
        print( '')
        print(f'Creating Mesh ...', end='', flush=True)
        self.last_time_check = time.time()
    
    def printAfterSetup(self):
        print(f' Done ({time.time() - self.last_time_check:.2f} s)')
        print( '')
        print( 'Populating A and b ...', end='', flush=True)
        self.last_time_check = time.time()

    def printAfterPopulation(self):
        print(f' Done ({time.time() - self.last_time_check:.2f} s)')
        print( '')
        print( 'Solving Inviscid ...', end='', flush=True)
        self.last_time_check = time.time()

    def printAfterInviscidSolve(self):
        print(f' Done ({time.time() - self.last_time_check:.2f} s)')
        print( '')
        print( 'Calculating Forces ...', end='', flush=True)
        self.last_time_check = time.time()
    
    def printForcesResults(self):
        print(f' Done ({time.time() - self.last_time_check:.2f} s)')
        print( '')
        print(f'Results Summary: ({"Converged" if self.converged else "Not Converged"})')
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
        print( 'Writing results to file ...', end='', flush=True)
        self.last_time_check = time.time()
    
    def printAfterWriting(self):
        print(f' Done ({time.time() - self.last_time_check:.2f} s)')
        print( '')
    
    def printBeforeVelField(self):
        print( 'Creating Visualisation ...', end='', flush=True)
        self.last_time_check = time.time()
    
    def printAfterVelField(self):
        print(f' Done ({time.time() - self.last_time_check:.2f} s)')
        print( '')
    
    def printFinal(self):
        print( 'Exiting ... Done')
        print( '')
        print(f'Total Runtime: {time.time() - self.start_time:.3f} s')
        print( '')
        print( '~' * 60)
        print( '')