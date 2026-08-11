# class for creating and storing all mesh values, including wake

import csv
import numpy as np
import matplotlib.pyplot as plt
import time

import Modules.ElementaryFlows as flows
import Modules.General as General

class Mesh:
    def __init__(self, config):
        # ============== Input Values ================ #
        self.config = config

        # ============== Load Geometry =============== #
        self.vertices = np.loadtxt(self.config.geo_file, skiprows=1)
        self.vertices *= self.config.chord

        # ============ Get Mesh Parameters ============ #
        self.getMeshParameters(vertices=self.vertices)


    def getMeshParameters(self, vertices):
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
        self.N = len(self.control_points)
        self.A = np.zeros((self.N+1, self.N+1)) # +1 for kutta condition
        self.b = np.zeros(self.N+1)

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


    # ====================== Solving ========================= #
    def solve(self):
        x = np.linalg.solve(self.A, self.b)
        self.source_strengths = x[:-1]
        self.vortex_strength = x[-1]


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
        
        fx = np.sum( - self.pressure * self.lengths * self.normals[:, 0] )
        fy = np.sum( - self.pressure * self.lengths * self.normals[:, 1] )

        return np.array([fx, fy])

    def viscousForce(self):
        return np.array([0, 0])

    def pressureMoment(self):

        r = self.control_points - self.config.ref_position
        forces = - self.pressure[:, None] * self.normals * self.lengths[:, None]

        return np.sum(r[:, 0] * forces[:, 1] - r[:, 1] * forces[:, 0])

    def viscousMoment(self):
        return 0

    # ====================== Running =========================== #
    def run(self):
        # Initial setup (happens automatically in __init__)
        if self.config.verbose:
            self.printInitial()

        # Solve etc.
        self.computeAandb()
        if self.config.verbose:
            self.printAfterSetup()

        self.solve()
        if self.config.verbose:
            self.printAfterSolving()

        # Solution Analysis
        self.calculatePressureOnPanels()
        self.calculateForcesAndMoments()
        if self.config.verbose:
            self.printForcesResults()

        # Output
        if self.config.write_to_file:
            ##### add writing here
            if self.config.verbose:
                self.printAfterWriting()
        
        # Visualisation
        if self.config.visualisation:
            self.visualisation = General.Visualisation(self.config, self)
            if self.config.verbose:
                self.printAfterVelField()
            self.visualisation.plot()
            ###### TODO: auto size arrows

        # Exit
        if self.config.verbose:
            self.printFinal()


    # ==================== Plotting ======================== #

    def plotGeometry(self, vertices=False, control_points=False, normals=False, tangents=False, gcs=False, vectors_percent_scale=100):
        plt.plot(self.vertices[:, 0], self.vertices[:, 1], color='black', linewidth=1)
        if vertices:
            plt.scatter(self.vertices[:, 0], self.vertices[:, 1], color='black', s=3)
        if control_points:
            self.plotControlPoints()
        if normals:
            self.plotNormals(size=vectors_percent_scale)
        if tangents:
            self.plotTangents(size=vectors_percent_scale)
        if gcs:
            self.plotGlobalCoordinateSystem(size=vectors_percent_scale)

    def plotControlPoints(self):
        plt.scatter(self.control_points[:, 0], self.control_points[:, 1], color='red', s=3)
    
    def plotNormals(self, size=100):
        plt.quiver(self.control_points[:, 0], self.control_points[:, 1], self.normals[:, 0], self.normals[:, 1], angles='xy', scale_units='xy', scale=100/size, width=0.005, color='g')

    def plotTangents(self, size=100):
        plt.quiver(self.control_points[:, 0], self.control_points[:, 1], self.tangents[:, 0], self.tangents[:, 1], angles='xy', scale_units='xy', scale=100/size, width=0.005, color='b')

    def plotGlobalCoordinateSystem(self, size=100):
        plt.quiver(0, 0, 1, 0, angles='xy', scale_units='xy', scale=100/size, width=0.005, color='r') # x
        plt.text(1.05 * size/100, 0, "x", color='r', ha='left', va='center', fontweight='bold')
        plt.quiver(0, 0, 0, 1, angles='xy', scale_units='xy', scale=100/size, width=0.005, color='b') # y
        plt.text(0, 1.05 * size/100, "y", color='b', ha='center', va='bottom', fontweight='bold')

    # =================== Printing ====================== #
    def printInitial(self):
        print( '')
        print( '2D Panel Method for Airfoils')
        print( '    Ryan Heskey')
        print( '')
        print( 'Input:')
        print(f'    Recieved Geometry File: {self.config.geo_file} ({len(self.vertices)} Vertices)')
        print(f'    Non-Dimensional Numbers: Re = {self.config.Re:.0f}, M = {self.config.M_inf}')
        print(f'    Pressure Calculation Mode: {self.config.pressure_calculation}')
        print( '')
        print( 'Setup:')
        print(f'    Creating Mesh ...', end='')
        self.meshing_start_time = time.time()
    
    def printAfterSetup(self):
        print(f' Done ({time.time() - self.meshing_start_time:.2f} s)')
        print( '')
        print( 'Solving ...', end='')
        self.solving_start_time = time.time()

    def printAfterSolving(self):
        print(f' Done ({time.time() - self.solving_start_time:.2f} s)')
        print( 'Calculating Forces ...', end='')
        self.forces_start_time = time.time()
    
    def printForcesResults(self):
        print(f' Done ({time.time() - self.forces_start_time:.2f} s)')
        print( '')
        print( 'Results Summary:')
        print(f'    Cx: {self.c_force[0]:.3f} ({self.total_force[0]:.1f} N/m)')
        print(f'    Cy: {self.c_force[1]:.3f} ({self.total_force[1]:.1f} N/m)')
        print( '')
        print(f'    Cd: {self.c_drag:.3f} ({self.drag:.2f} N/m)')
        print(f'    Cl: {self.c_lift:.3f} ({self.lift:.2f} N/m)')
        print( '')
        print(f'    Cm: {self.c_moment:.3f} ({self.total_moment:.2f} N.m/m) around (x, y) = ({self.config.ref_position[0]}, {self.config.ref_position[1]}) m')
        print( '')
        if self.config.write_to_file:
            print( 'Writing to File ...', end='')
            self.write_start_time = time.time()
        elif self.config.visualisation:
            print( 'Creating Velocity Field for Visualisation ...', end='')
            self.visual_start_time = time.time()
    
    def printAfterWriting(self):
        print(f' Done ({time.time() - self.write_start_time:.2f} s)')
        print( '')
        if self.config.visualisation:
            print( 'Creating Velocity Field for Visualisation ...', end='')
            self.visual_start_time = time.time()
    
    def printAfterVelField(self):
        print(f' Done ({time.time() - self.visual_start_time:.2f} s)')
        print( '')
    
    def printFinal(self):
        print(f'Exiting ... Done')
        print( '')