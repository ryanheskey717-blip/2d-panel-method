"""
Elementary Flows
================

This module provides the core mathematical functions for evaluating 2D potential flow 
components. It includes induced velocity calculations for constant-strength source 
distributions and vortex sheets used to solve the inviscid flow field.
"""
import numpy as np

def getConstantSourcePanelVelocity(vertex_1, vertex_2, sample_point, strength):
    r"""Calculates the velocity contribution at a sample point from a 2D source panel.

    This function computes the induced velocity `(u, v)` from a constant-strength 
    source panel defined between two vertices using standard 2D potential flow equations.
    
    Args:
        vertex_1 (tuple or ndarray): The `(x, y)` coordinates of the panel's first vertex.
        vertex_2 (tuple or ndarray): The `(x, y)` coordinates of the panel's second vertex.
        sample_point (tuple or ndarray): The `(x, y)` coordinates where the velocity is evaluated.
        strength (float): The constant source strength $\sigma$ of the panel.

    Returns:
        tuple[float, float]: The induced velocity `(u, v)` in the global $x$ and $y$ directions.
    """
    L = np.sqrt( ( vertex_2[0] - vertex_1[0])**2 + (vertex_2[1] - vertex_1[1])**2 ) # length of panel
    phi = np.atan2( vertex_2[1]-vertex_1[1], vertex_2[0]-vertex_1[0]) # rotation angle of the panel

    dx = sample_point[0] - vertex_1[0]
    dy = sample_point[1] - vertex_1[1]

    xl = dx * np.cos(phi) + dy * np.sin(phi)
    yl = -dx * np.sin(phi) + dy * np.cos(phi)

    r1 = np.sqrt(xl**2 + yl**2) # distance from vertex 1
    r2 = np.sqrt((xl-L)**2 + yl**2) # distance from vertex 2

    r1 = np.clip(r1, 1e-12, None) # make sure there is no divide by zero
    r2 = np.clip(r2, 1e-12, None)

    theta1 = np.atan2(yl, xl)
    theta2 = np.atan2(yl, xl - L)

    beta = theta2 - theta1 # angular width of panel from sample point
    beta = (beta + np.pi) % (2*np.pi) - np.pi # wrap into (-pi, pi]

    ul = - strength / (2 * np.pi) * np.log( r2 / r1 ) # local horizontal velocity (along panel)
    vl = strength / (2 * np.pi) * beta # local vertical velocity (normal to panel)

    u = ul * np.cos(phi) - vl * np.sin(phi)
    v = ul * np.sin(phi) + vl * np.cos(phi)
    
    return u, v

def getConstantSourcePanelsVelocity(mesh, sample_point):
    """Calculates the contribution of all source panels to the velocity at an arbitrary point.

    Args:
        mesh (object): Instance of the Mesh class.
        sample_point (ndarray): (x, y) numpy array of coordinates of sampling point.
    
    Returns:
        tuple[float, float]: Resultant velocity $(u_{total}, v_{total})$ in the horizontal (x) and vertical (y) directions
    """
    ut = 0; vt = 0
    for i in range(mesh.N):
        u, v = getConstantSourcePanelVelocity(mesh.vertices[i], mesh.vertices[i+1], sample_point, mesh.source_strengths[i])
        ut += u
        vt += v
    return ut, vt

def getConstantVortexPanelVelocity(vertex_1, vertex_2, sample_point, strength):
    r"""Calculates the velocity contribution at a sample point from a 2D vortex panel.

    This function computes the induced velocity `(u, v)` from a constant-strength 
    vortex panel defined between two vertices using standard 2D potential flow equations.
    
    Args:
        vertex_1 (tuple or ndarray): The `(x, y)` coordinates of the panel's first vertex.
        vertex_2 (tuple or ndarray): The `(x, y)` coordinates of the panel's second vertex.
        sample_point (tuple or ndarray): The `(x, y)` coordinates where the velocity is evaluated.
        strength (float): The constant vortex strength $\gamma$ of the panel.

    Returns:
        tuple[float, float]: The induced velocity `(u, v)` in the global $x$ and $y$ directions.
    """
    L = np.sqrt( ( vertex_2[0] - vertex_1[0])**2 + (vertex_2[1] - vertex_1[1])**2 ) # length of panel
    phi = np.atan2( vertex_2[1]-vertex_1[1], vertex_2[0]-vertex_1[0]) # rotation angle of the panel

    dx = sample_point[0] - vertex_1[0]
    dy = sample_point[1] - vertex_1[1]

    xl = dx * np.cos(phi) + dy * np.sin(phi)
    yl = -dx * np.sin(phi) + dy * np.cos(phi)

    r1 = np.sqrt(xl**2 + yl**2) # distance from vertex 1
    r2 = np.sqrt((xl-L)**2 + yl**2) # distance from vertex 2

    r1 = np.clip(r1, 1e-12, None) # make sure there is no divide by zero
    r2 = np.clip(r2, 1e-12, None)

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

def getConstantVortexPanelsVelocity(mesh, sample_point, strength=None):
    r"""Calculates the contribution of all vortex panels to the velocity at an arbitrary point.

    Args:
        mesh (object): Instance of the Mesh class containing panel geometries.
        sample_point (ndarray): $(x, y)$ NumPy array of coordinates for the sampling point.
        strength (float, optional): Circulation strength $\gamma$. If not specified, assumed to be contained in mesh.

    Returns:
        tuple[float, float]: Resultant velocity $(u_{total}, v_{total})$ in the horizontal (x) and vertical (y) directions
    """
    ut = 0; vt = 0
    for i in range(mesh.N):
        if strength is None: # used after panel method has been run
            u, v = getConstantVortexPanelVelocity(mesh.vertices[i], mesh.vertices[i+1], sample_point, mesh.vortex_strength)
        else:
            u, v = getConstantVortexPanelVelocity(mesh.vertices[i], mesh.vertices[i+1], sample_point, strength)
        ut += u
        vt += v
    return ut, vt