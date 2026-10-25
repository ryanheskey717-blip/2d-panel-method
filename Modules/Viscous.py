r"""
Viscous Coupling Module
=======================

Provides boundary layer calculations and viscous-inviscid coupling routines 
using the transpiration method. Includes laminar Blasius models for 
displacement thickness and skin friction calculations.
"""
import numpy as np

# ================= Laminar Blasius Method =================== #

#### TODO: update blasius to instead be distance along panel instead of just x coord

def blasiusDeltaStar(mesh): # at control points
    r"""Calculates the laminar displacement thickness at panel control points.

    Uses the Blasius flat plate boundary layer solution to compute the displacement thickness 
    $\delta^*$ based on the local Reynolds number $Re_x$.

    Args:
        mesh (Mesh): The mesh object containing geometry, control points, and configuration data.

    Returns:
        ndarray: Array of displacement thickness values for each panel.
    """
    delta_star = np.zeros(mesh.N)

    # loop over every control point (middle of each panel) and calc d* here.
    for i in range(mesh.N):
        Re_x = mesh.config.V_inf * (mesh.control_points[i, 0]) / mesh.config.nu_inf
        if Re_x == 0.0:
            delta_star[i] = 0
        else:
            delta_star[i] = 1.72 * mesh.control_points[i, 0] / np.sqrt(Re_x) # flat plate approx.
    
    return delta_star

def blasiusCf(mesh): # at control points
    r"""Calculates the laminar skin friction coefficient at panel control points.

    Uses the Blasius flat plate solution to evaluate the local skin friction 
    $C_f$ based on the local Reynolds number.

    Args:
        mesh (Mesh): The mesh object containing geometry and freestream conditions.

    Returns:
        ndarray: Array of local skin friction coefficients.
    """
    c_f = np.zeros(mesh.N)

    # find cf at each control point (middle of each panel)
    for i in range(mesh.N):
        Re_x = mesh.config.V_inf * (mesh.control_points[i, 0]) / mesh.config.nu_inf
        if Re_x == 0.0:
            c_f[i] = 0
        else:
            c_f[i] = 0.664 / np.sqrt(mesh.config.V_inf * (mesh.control_points[i, 0]) / mesh.config.nu_inf) # flat plate approx.
    
    return c_f

# ====================== General ============================ #
def findStagnationPanel(mesh):
    r"""Locates the stagnation point on the airfoil surface.

    Scans the discretized panels to identify the index where the tangential 
    velocity changes sign, indicating the flow stagnation point (needs work).

    Args:
        mesh (Mesh): The mesh object containing tangential velocity data.

    Returns:
        int: The index `i_stag` of the panel containing the stagnation point.
    """
    mesh.i_stag = 0
    for i in range(mesh.N):
        if mesh.velocity_tangent_at_vertices[i+1] * mesh.velocity_tangent_at_vertices[i] < 0:
            mesh.i_stag = i # definitely not a good way of doing it
            break
    return mesh.i_stag

# ================== Called from mesh class ================== #
def updateDisplacementThickness(mesh):
    r"""Updates the displacement thickness distribution for the current flow state.

    Evaluates the boundary layer model specified in the solver configuration (falling 
    back to Blasius if the requested method is unsupported) and interpolates the 
    control point values to the panel vertices.

    Args:
        mesh (Mesh): The mesh object containing configuration settings and geometric data.
    """
    # choose which method to use based on input
    if mesh.config.boundary_layer_calculation == 'blasius':
        mesh.delta_star = blasiusDeltaStar(mesh)
    else:
        print(f'Warning: {mesh.config.boundary_layer_calculation} not yet supported, switching to Blasius.')
        mesh.delta_star = blasiusDeltaStar(mesh) #### TODO: update with any new method

    # interpolate to vertices
    mesh.delta_star_at_vertices = mesh.interpControlPointsToVertices(mesh.delta_star)

def updateTranspirationVelocity(mesh):
    r"""Calculates the normal 'blowing' velocity to simulate the boundary layer effect.

    Locates the stagnation point and marches upstream along the upper and lower 
    surfaces. Computes the transpiration velocity using a centered finite difference 
    approximation of $V_n = \frac{d}{ds}(|U_e| \delta^*)$. 

    Args:
        mesh (Mesh): The mesh object to update with the computed `transpiration_velocity` array.
    """
    # find stagnation panel
    i_stag = findStagnationPanel(mesh)
    
    # march upstream along top and bottom starting from stagnation point
        # using centred finite difference to get transpiration velocity
    trans_vel = np.zeros(mesh.N)
    for i in np.arange(i_stag, -1, -1): # top surface
        trans_vel[i] = ((abs(mesh.velocity_tangent_at_vertices[i]) * mesh.delta_star_at_vertices[i]) - (abs(mesh.velocity_tangent_at_vertices[i+1]) * mesh.delta_star_at_vertices[i+1])) / mesh.lengths[i]
    for i in range(i_stag+1, mesh.N): # bottom surface
        trans_vel[i] = ((abs(mesh.velocity_tangent_at_vertices[i+1]) * mesh.delta_star_at_vertices[i+1]) - (abs(mesh.velocity_tangent_at_vertices[i]) * mesh.delta_star_at_vertices[i])) / mesh.lengths[i]
    
    # set to 0 for first and last panel (as it gives a falsely large value due to interpolation of d* at first and last vertex)
    trans_vel[0] = 0.0
    trans_vel[-1] = 0.0
    mesh.transpiration_velocity = trans_vel

def getSkinFrictionCoefficient(mesh):
    r"""Retrieves the skin friction coefficient distribution over the airfoil.

    Evaluates the boundary layer method specified in the configuration 
    to obtain local skin friction values[cite: 7].

    Args:
        mesh (Mesh): The mesh object containing configuration settings.

    Returns:
        ndarray: Array of local skin friction coefficients for each panel.
    """
    # decide which method to use
    if mesh.config.boundary_layer_calculation == 'blasius':
        c_f = blasiusCf(mesh)
    else:
        print(f'Warning: {mesh.config.boundary_layer_calculation} not yet supported, switching to Blasius.')
        c_f = blasiusCf(mesh) #### TODO: update with any new method
    return c_f