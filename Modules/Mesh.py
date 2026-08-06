# class for creating and storing all mesh values, including wake

import csv
import numpy as np
import matplotlib.pyplot as plt

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
        self.A = np.zeros((self.N, self.N))
        self.b = np.zeros(self.N)

    
    # ================ Elementary flows =================== #
    def getPointSourceVelocity(self, source_point, sample_point, strength):
        distance = np.sqrt((sample_point[0] - source_point[0])**2 + (sample_point[1] - source_point[1])**2)
        u = strength/(2*np.pi) * (sample_point[0] - source_point[0])/distance
        v = strength/(2*np.pi) * (sample_point[1] - source_point[1])/distance
        return u, v
    
    def getPointSourcesVelocity(self, sample_point):
        ut = 0; vt = 0
        for i in range(self.N):
            u, v = self.getPointSourceVelocity(self.control_points[i], sample_point, self.source_strengths[i])
            ut += u
            vt += v
        return ut, vt
    
    def getConstantSourcePanelVelocity(self, vertex_1, vertex_2, sample_point, strength):

        L = np.sqrt( ( vertex_2[0] - vertex_1[0])**2 + (vertex_2[1] - vertex_1[1])**2 ) # length of panel
        phi = np.atan2( vertex_2[1]-vertex_1[1], vertex_2[0]-vertex_1[0]) # rotation angle of the panel

        dx = sample_point[0] - vertex_1[0]
        dy = sample_point[1] - vertex_1[1]

        xl = dx * np.cos(phi) + dy * np.sin(phi)
        yl = -dx * np.sin(phi) + dy * np.cos(phi)

        r1 = np.sqrt(xl**2 + yl**2) # distance from vertex 1
        r2 = np.sqrt((xl-L)**2 + yl**2) # distance from vertex 2

        theta1 = np.atan2(yl, xl)
        theta2 = np.atan2(yl, xl - L)

        beta = theta2 - theta1 # angular width of panel from sample point
        beta = (beta + np.pi) % (2*np.pi) - np.pi # wrap into (-pi, pi]

        ul = strength / (2 * np.pi) * np.log( r1 / r2 ) # local horizontal velocity (along panel)
        vl = strength / (2 * np.pi) * beta # local vertical velocity (normal to panel)

        u = ul * np.cos(phi) - vl * np.sin(phi)
        v = ul * np.sin(phi) + vl * np.cos(phi)
        
        return u, v
    
    def getConstantSourcePanelsVelocity(self, sample_point):
        ut = 0; vt = 0
        for i in range(self.N):
            u, v = self.getConstantSourcePanelVelocity(self.vertices[i], self.vertices[i+1], sample_point, self.source_strengths[i])
            ut += u
            vt += v
        return ut, vt

    # =================== Preconditioning =================== #
    def computeAandb(self):
        for i in range(self.N): # control point
            for j in range(self.N): # panel
                if i == j:
                    self.A[i, j] = 0.5 # at self control point, velocity == 0
                else:
                    u, v = self.getConstantSourcePanelVelocity(self.vertices[j], self.vertices[j+1], self.control_points[i], strength=1.0)
                    self.A[i, j] = np.dot(self.normals[i], np.array([u, v]))
        
        for i in range(self.N): # control points
            self.b[i] = - np.dot(self.config.V_inf_vec, self.normals[i])


    # ====================== Solving ========================= #
    def solve(self):
        self.source_strengths = np.linalg.solve(self.A, self.b)

        if False: #TODO: DELETE #######################
            print('A:', self.A)
            print('b:', self.b)
            print('sig:', self.source_strengths)


    # ================= Flow Properties =================== #

    def calculateTangentialVelocityAtControlPoints(self):
        self.velocity_tangent_at_control_points = np.zeros(self.N)
        for i in range(self.N): # control point
            u_total = self.config.V_inf_vec[0] # initialise at freestream velocity
            v_total = self.config.V_inf_vec[1]
            
            for j in range(self.N): # panel
                u, v = self.getConstantSourcePanelVelocity(self.vertices[j], self.vertices[j+1], self.control_points[i], strength=self.source_strengths[j])
                u_total += u
                v_total += v
            self.velocity_tangent_at_control_points[i] = np.dot(np.array([u_total, v_total]), self.tangents[i])

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
        print('\n', self.total_force, '[N/m]')
        print(self.c_force, '[]')

        print('\n', self.total_moment, '[N.m/m]')
        print(self.c_moment, '[]')

    def pressureForce(self): # midpoint integrate for forces (as only know pressure at control points i.e. centres)
        
        fx = np.sum( - self.pressure * self.lengths * self.normals[:, 0])
        fy = np.sum( - self.pressure * self.lengths * self.normals[:, 1])

        return np.array([fx, fy])

    def viscousForce(self):
        return np.array([0, 0])

    def pressureMoment(self):

        r = self.control_points - self.config.ref_position
        forces = - self.pressure[:, None] * self.normals * self.lengths[:, None]

        return np.sum(r[:, 0] * forces[:, 1] - r[:, 1] * forces[:, 0])

    def viscousMoment(self):
        return 0


    # ==================== Plotting ======================== #

    def plotGeometry(self, vertices=False, control_points=False, normals=False, tangents=False, vectors_percent_scale=100):
        plt.plot(self.vertices[:, 0], self.vertices[:, 1], color='black', linewidth=1)
        if vertices:
            plt.scatter(self.vertices[:, 0], self.vertices[:, 1], color='black', s=3)
        if control_points:
            self.plotControlPoints()
        if normals:
            self.plotNormals(size=vectors_percent_scale)
        if tangents:
            self.plotTangents(size=vectors_percent_scale)

    def plotControlPoints(self):
        plt.scatter(self.control_points[:, 0], self.control_points[:, 1], color='red', s=3)
    
    def plotNormals(self, size=100):
        plt.quiver(self.control_points[:, 0], self.control_points[:, 1], self.normals[:, 0], self.normals[:, 1], angles='xy', scale_units='xy', scale=100/size, width=0.005, color='g')

    def plotTangents(self, size=100):
        plt.quiver(self.control_points[:, 0], self.control_points[:, 1], self.tangents[:, 0], self.tangents[:, 1], angles='xy', scale_units='xy', scale=100/size, width=0.005, color='b')