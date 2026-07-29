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

    # =================== Preconditioning =================== #
    def computeAandb(self):
        for i in range(len(self.control_points)):
            for j in range(len(self.control_points)):
                if i == j:
                    self.A[j, i] = 0.5 # at self control point, velocity == 0
                else:
                    control_point = np.array([self.control_points[j, 0], self.control_points[j, 1]]) + self.normals[i] * self.control_point_offset
                    u, v = self.getPointSourceVelocity((self.control_points[j, 0], self.control_points[j, 1]), control_point, strength=1.0)
                    self.A[j, i] = np.dot(self.normals[i], np.array([u, v]))
        
        for i in range(len(self.control_points)):
            self.b[i] = - np.dot(self.V_inf_vec, self.normals[i])

        if False: ########################
            print('A:', self.A)
            print('b:', self.b)


    # ====================== Solving ========================= #
    def solve(self):
        self.source_strengths = np.linalg.solve(self.A, self.b)

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