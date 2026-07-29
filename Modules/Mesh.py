# class for creating and storing all mesh values, including wake

import csv
import numpy as np
import matplotlib.pyplot as plt

class Mesh:
    def __init__(self, config):
        # ============== Important values from config ================ #
        self.verbose = config.verbose

        # ==================== Load geometry ======================== #
        self.vertices = np.loadtxt(config.geo_file, skiprows=1)
        self.vertices *= config.chord

        # ============= Calculate control points and normals ============== #
        self.control_points = []
        for i in range(len(self.vertices) - 1):
            self.control_points.append([
                (self.vertices[i, 0] + self.vertices[i+1, 0]) / 2, 
                (self.vertices[i, 1] + self.vertices[i+1, 1]) / 2
            ])
        self.control_points = np.array(self.control_points)

        self.normals = []
        for i in range(len(self.vertices) - 1):
            self.normals.append([
                (self.vertices[i+1, 1] - self.vertices[i, 1]),
                (self.vertices[i, 0] - self.vertices[i+1, 0]),
            ])
        self.normals = np.array(self.normals)
        self.normals /= np.linalg.norm(self.normals, axis=1, keepdims=True) # normalise

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
        for i in range(len(self.control_points)):
            distance = np.sqrt((sample_point[0] - self.control_points[i, 0])**2 + (sample_point[1] - self.control_points[i, 1])**2)
            u = self.source_strengths[i]/(2*np.pi) * (sample_point[0] - self.control_points[i, 0])/distance
            v = self.source_strengths[i]/(2*np.pi) * (sample_point[1] - self.control_points[i, 0])/distance
        return u, v

    # =================== Preconditioning =================== #
    def computeAandb(self):
        for i in range(len(self.control_points)):
            for j in range(len(self.control_points)):
                if i == j:
                    self.A[j, i] = 0 # at self control point, velocity == 0
                else:
                    u, v = self.getPointSourceVelocity((self.control_points[j, 0], self.control_points[j, 1]), (self.control_points[i, 0], self.control_points[i, 1]), strength=1.0)
                    self.A[j, i] = np.dot(self.normals[i], np.array([u, v]))
        
        for i in range(len(self.control_points)):
            self.b[i] = - np.dot(self.V_inf_vec, self.normals[i])

        if self.verbose:
            print('A:', self.A)
            print('b:', self.b)


    # ====================== Solving ========================= #
    def solve(self):
        self.source_strengths = np.linalg.solve(self.A, self.b)

    # ==================== Plotting ======================== #

    def plotGeometry(self, points=True):
        plt.plot(self.vertices[:, 0], self.vertices[:, 1], color='black', linewidth=1)
        if points:
            plt.scatter(self.vertices[:, 0], self.vertices[:, 1], color='black', s=3)

    def plotControlPoints(self):
        plt.scatter(self.control_points[:, 0], self.control_points[:, 1], color='red', s=3)
    
    def plotNormals(self, size=100):
        mult = size/100
        plt.quiver(self.control_points[:, 0], self.control_points[:, 1], self.normals[:, 0]*mult, self.normals[:, 1]*mult, angles='xy', scale_units='xy', scale=1, color='g')