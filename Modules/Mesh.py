# class for creating and storing all mesh values, including wake

import csv
import numpy as np
import matplotlib.pyplot as plt

class Mesh:
    def __init__(self, config):
        # ============== Important values from config ================ #
        self.verbose = config.verbose
        self.control_point_offset = config.control_point_offset

        # ==================== Load geometry ======================== #
        self.vertices = np.loadtxt(config.geo_file, skiprows=1)
        self.vertices *= config.chord

        # ============= Calculate control points, normals and tangents ============== #
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
        self.tangents /= np.linalg.norm(self.tangents, axis=1, keepdims=True) # normalise

        self.control_points = []
        for i in range(len(self.vertices) - 1):
            self.control_points.append([
                (self.vertices[i, 0] + self.vertices[i+1, 0]) / 2, 
                (self.vertices[i, 1] + self.vertices[i+1, 1]) / 2
            ])
        self.control_points = np.array(self.control_points)

        # ================= Preallocate Arrays ==================== #
        self.A = np.zeros((len(self.control_points), len(self.control_points)))
        self.b = np.zeros(len(self.control_points))

        # add freestream
        self.V_inf_vec = config.V_inf_vec

    
    # ================ Elementary flows =================== #
    def getPointSourceVelocity(self, source_point, sample_point, strength):
        distance = np.sqrt((sample_point[0] - source_point[0])**2 + (sample_point[1] - source_point[1])**2)
        u = strength/(2*np.pi) * (sample_point[0] - source_point[0])/distance
        v = strength/(2*np.pi) * (sample_point[1] - source_point[1])/distance
        return u, v
    
    def getPointSourcesVelocity(self, sample_point):
        ut = 0; vt = 0
        for i in range(len(self.control_points)):
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
        for i in range(len(self.control_points)):
            u, v = self.getConstantSourcePanelVelocity(self.vertices[i], self.vertices[i+1], sample_point, self.source_strengths[i])
            ut += u
            vt += v
        return ut, vt

    # =================== Preconditioning =================== #
    def computeAandb(self):
        for i in range(len(self.control_points)): # control point
            control_point = self.control_points[i] + self.normals[i] * self.control_point_offset
            for j in range(len(self.control_points)): # panel
                if i == j:
                    self.A[i, j] = 0.5 # at self control point, velocity == 0
                else:
                    u, v = self.getConstantSourcePanelVelocity(self.vertices[j], self.vertices[j+1], control_point, strength=1.0)
                    self.A[i, j] = np.dot(self.normals[i], np.array([u, v]))
        
        for i in range(len(self.control_points)): # control points
            self.b[i] = - np.dot(self.V_inf_vec, self.normals[i])


    # ====================== Solving ========================= #
    def solve(self):
        self.source_strengths = np.linalg.solve(self.A, self.b)

        if False: #TODO: DELETE #######################
            print('A:', self.A)
            print('b:', self.b)
            print('sig:', self.source_strengths)

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