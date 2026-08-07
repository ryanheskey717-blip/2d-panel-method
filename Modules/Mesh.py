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
        self.A = np.zeros((self.N+1, self.N+1)) # +1 for kutta condition
        self.b = np.zeros(self.N+1)

    
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

        ul = - strength / (2 * np.pi) * np.log( r2 / r1 ) # local horizontal velocity (along panel)
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

    def getConstantDoubletPanelVelocity(self, vertex_1, vertex_2, sample_point, strength):

        L = np.sqrt( ( vertex_2[0] - vertex_1[0])**2 + (vertex_2[1] - vertex_1[1])**2 ) # length of panel
        phi = np.atan2( vertex_2[1]-vertex_1[1], vertex_2[0]-vertex_1[0]) # rotation angle of the panel

        dx = sample_point[0] - vertex_1[0]
        dy = sample_point[1] - vertex_1[1]

        xl = dx * np.cos(phi) + dy * np.sin(phi)
        yl = -dx * np.sin(phi) + dy * np.cos(phi)

        r1_2 = xl**2 + yl**2 # distance from vertex 1
        r2_2 = (xl-L)**2 + yl**2 # distance from vertex 2

        ul = strength / (2 * np.pi) * yl * (1/r1_2 - 1/r2_2) # local horizontal velocity (along panel)
        vl = - strength / (2 * np.pi) * (xl/r1_2 - (xl-L)/r2_2) # local vertical velocity (normal to panel)

        u = ul * np.cos(phi) - vl * np.sin(phi)
        v = ul * np.sin(phi) + vl * np.cos(phi)

        return u, v

    def getConstantDoubletPanelsVelocity(self, sample_point, strength=1.0):
        ut = 0; vt = 0
        for i in range(self.N):
            u, v = self.getConstantVortexPanelVelocity(self.vertices[i], self.vertices[i+1], sample_point, strength)
            ut += u
            vt += v
        return ut, vt

    def getConstantVortexPanelVelocity(self, vertex_1, vertex_2, sample_point, strength):
    
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

        ul = 0; vl = 0
        if xl >= 0 and xl <= L:
            if yl <= 1e-5 and yl > 0:
                ul = - strength / 2
            elif yl >= -1e-5 and yl < 0:
                ul = strength / 2
            elif yl == 0:
                ul = 0
            else:
                ul = - strength / (2 * np.pi) * beta # local horizontal velocity (along panel)
            vl = strength / (2 * np.pi) * np.log( r1 / r2 ) # local vertical velocity (normal to panel)
        else:
            ul = - strength / (2 * np.pi) * beta # local horizontal velocity (along panel)
            vl = strength / (2 * np.pi) * np.log( r1 / r2 ) # local vertical velocity (normal to panel)

        u = ul * np.cos(phi) - vl * np.sin(phi)
        v = ul * np.sin(phi) + vl * np.cos(phi)
        
        return u, v
    
    def getConstantVortexPanelsVelocity(self, sample_point, strength=None):
        ut = 0; vt = 0
        for i in range(self.N):
            if strength is None:
                u, v = self.getConstantVortexPanelVelocity(self.vertices[i], self.vertices[i+1], sample_point, self.vortex_strength)
            else:
                u, v = self.getConstantVortexPanelVelocity(self.vertices[i], self.vertices[i+1], sample_point, strength)
            ut += u
            vt += v
        return ut, vt

    # =================== Preconditioning =================== #
    def computeAandb(self):

        # section of A relating to no-penetration boundary condition
        for i in range(self.N): # control point
            for j in range(self.N + 1): # panel + influence from constant strength vortex sheet
                if j == self.N: # vortex influence
                    u, v = self.getConstantVortexPanelsVelocity(self.control_points[i], strength=1.0)
                    self.A[i, j] = np.dot(self.normals[i], np.array([u, v]))
                else:
                    if i == j: # source self-influence
                        self.A[i, j] = 0.5
                    else:
                        u, v = self.getConstantSourcePanelVelocity(self.vertices[j], self.vertices[j+1], self.control_points[i], strength=1.0)
                        self.A[i, j] = np.dot(self.normals[i], np.array([u, v]))

        # section of A relating to kutta condition (these find tangential influences)
        for j in range(self.N + 1): # panel + influence from constant strength vortex sheet
            if j == self.N: # vortex influence
                u1, v1 = self.getConstantVortexPanelsVelocity(self.control_points[0], strength=1.0)  # at upper TE control point
                u2, v2 = self.getConstantVortexPanelsVelocity(self.control_points[-1], strength=1.0) # at lower TE control point
                self.A[-1, j] = (np.dot(self.tangents[0], np.array([u1, v1]))) + (np.dot(self.tangents[-1], np.array([u2, v2])))
            else:
                u1, v1 = self.getConstantSourcePanelVelocity(self.vertices[j], self.vertices[j+1], self.control_points[0], strength=1.0)  # at upper TE control point
                u2, v2 = self.getConstantSourcePanelVelocity(self.vertices[j], self.vertices[j+1], self.control_points[-1], strength=1.0) # at lower TE control point
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
            total_vel = self.config.V_inf_vec + self.getConstantSourcePanelsVelocity(self.control_points[i]) + self.getConstantVortexPanelsVelocity(self.control_points[i])
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

        print(f'\nFx: {self.total_force[0]:.2f} N/m | Fy: {self.total_force[1]:.2f} N/m')
        print(f'Cx: {self.c_force[0]:.2f}      | Cy: {self.c_force[1]:.2f}')

        lift = np.dot(self.total_force, np.array([-np.sin(self.config.alpha),np.cos(self.config.alpha)]))
        drag = np.dot(self.total_force, np.array([np.cos(self.config.alpha), np.sin(self.config.alpha)]))
        c_lift = np.dot(self.c_force, np.array([-np.sin(self.config.alpha), np.cos(self.config.alpha)]))
        c_drag = np.dot(self.c_force, np.array([np.cos(self.config.alpha), np.sin(self.config.alpha)]))

        print(f'\nFd: {drag:.2f} N/m   | Fl: {lift:.2f} N/m')
        print(f'Cd: {c_drag:.2f}       | Cl: {c_lift:.2f}')

        print(f'\nM: {self.total_moment:.2f} N.m/m  | Cm: {self.c_moment:.2f}')

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