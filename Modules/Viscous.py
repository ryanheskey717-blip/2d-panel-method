import numpy as np

# ================= Laminar Blasius Method =================== #

#### TODO: update blasius to instead be distance along panel instead of just x coord

def blasiusDeltaStar(mesh): # at control points
    delta_star = np.zeros(mesh.N)

    for i in range(mesh.N):
        Re_x = mesh.config.V_inf * (mesh.control_points[i, 0]) / mesh.config.nu_inf
        if Re_x == 0.0:
            delta_star[i] = 0
        else:
            delta_star[i] = 1.72 * mesh.control_points[i, 0] / np.sqrt(Re_x)
    
    return delta_star

def blasiusCf(mesh): # at control points
    c_f = np.zeros(mesh.N)

    for i in range(mesh.N):
        Re_x = mesh.config.V_inf * (mesh.control_points[i, 0]) / mesh.config.nu_inf
        if Re_x == 0.0:
            c_f[i] = 0
        else:
            c_f[i] = 0.664 / np.sqrt(mesh.config.V_inf * (mesh.control_points[i, 0]) / mesh.config.nu_inf)
    
    return c_f

# ====================== General ============================ #
def findStagnationPanel(mesh):
    mesh.i_stag = 0
    for i in range(mesh.N):
        if mesh.velocity_tangent_at_vertices[i+1] * mesh.velocity_tangent_at_vertices[i] < 0:
            mesh.i_stag = i
            break
    return mesh.i_stag

# ================== Called from mesh class ================== #
def updateDisplacementThickness(mesh):
    if mesh.config.boundary_layer_calculation == 'blasius':
        mesh.delta_star = blasiusDeltaStar(mesh)
    else:
        print(f'Warning: {mesh.config.boundary_layer_calculation} not yet supported, switching to Blasius.')
        mesh.delta_star = blasiusDeltaStar(mesh) #### TODO: update with any new method

    # interpolate to vertices
    mesh.delta_star_at_vertices = mesh.interpControlPointsToVertices(mesh.delta_star)

def updateTranspirationVelocity(mesh):

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
    if mesh.config.boundary_layer_calculation == 'blasius':
        c_f = blasiusCf(mesh)
    else:
        print(f'Warning: {mesh.config.boundary_layer_calculation} not yet supported, switching to Blasius.')
        c_f = blasiusCf(mesh) #### TODO: update with any new method
    return c_f