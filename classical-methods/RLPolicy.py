# Perspective transformation
# Discrete points to curve >> To lane center path which will be the reference path
# MPC and pure-pursuit based on the refernce path

import time

import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy import savetxt
from numpy.linalg import inv
#from matplotlib.animation import FuncAnimation

from Image2Waypoints import Image2Waypoints

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


import casadi as ca


from scipy.interpolate import interp1d

def interpolate_waypoints_to_horizon(waypoints_meters, N):
    """Interpolate waypoints to get evenly spaced reference for NMPC."""
    x_path = waypoints_meters[:, 0]
    y_path = waypoints_meters[:, 1]

    # Compute cumulative distances along the path
    distances = np.insert(np.cumsum(np.linalg.norm(np.diff(waypoints_meters, axis=0), axis=1)), 0, 0)
    total_length = distances[-1]

    # Create interpolators
    fx = interp1d(distances, x_path, kind='linear', fill_value='extrapolate')
    fy = interp1d(distances, y_path, kind='linear', fill_value='extrapolate')

    # Interpolate along arc-length
    interp_dists = np.linspace(0, total_length, N + 1)
    x_ref = fx(interp_dists)
    y_ref = fy(interp_dists)

    # Approximate heading (theta) using gradient
    dx = np.gradient(x_ref)
    dy = np.gradient(y_ref)
    theta_ref = np.arctan2(dy, dx)

    return np.vstack([x_ref, y_ref, theta_ref])  # shape: (3, N+1)



import casadi as ca

def create_nmpc_solver(N=10, dt=0.05,
                       Q_weights=[0.1, 0.1, 0.05],
                       R_weights=[0.1, 0.1],
                       v_des=0.75,
                       alpha=0.01):
    """
    Creates a nonlinear MPC solver for a differential drive robot using CasADi,
    including soft tracking of a desired linear velocity.

    Parameters:
        N : int
            Horizon length.
        dt : float
            Timestep (s).
        Q_weights : list of 3 floats
            State cost weights [x, y, theta].
        R_weights : list of 2 floats
            Control cost weights [v, omega].
        v_des : float
            Desired linear velocity (m/s).
        alpha : float
            Penalty weight on velocity tracking (soft constraint).

    Returns:
        solver : CasADi solver object
        solver_vars : dict with 'X', 'U', 'X_ref', and dimensions
    """

    # Symbolic variables
    x = ca.SX.sym('x')
    y = ca.SX.sym('y')
    theta = ca.SX.sym('theta')
    v = ca.SX.sym('v')
    omega = ca.SX.sym('omega')

    states = ca.vertcat(x, y, theta)
    controls = ca.vertcat(v, omega)

    # Dynamics
    rhs = ca.vertcat(
        v * ca.cos(theta),
        v * ca.sin(theta),
        omega
    )
    f = ca.Function('f', [states, controls], [rhs])

    # Optimization variables
    X = ca.SX.sym('X', 3, N+1)
    U = ca.SX.sym('U', 2, N)
    X_ref = ca.SX.sym('X_ref', 3, N+1)

    # Cost function weights
    Q = ca.diag(ca.SX(Q_weights))
    R = ca.diag(ca.SX(R_weights))

    # Objective and constraints
    obj = 0
    g = []

    g.append(X[:, 0] - X_ref[:, 0])  # initial constraint

    for k in range(N):
        x_next = X[:, k] + dt * f(X[:, k], U[:, k])
        g.append(X[:, k+1] - x_next)

        # Path tracking cost
        obj += ca.mtimes([(X[:, k] - X_ref[:, k]).T, Q, (X[:, k] - X_ref[:, k])])
        obj += ca.mtimes([U[:, k].T, R, U[:, k]])

        # Soft velocity tracking penalty on v
        obj += alpha * (U[0, k] - v_des) ** 2

    g = ca.vertcat(*g)

    # Flatten decision variables
    decision_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
    params = ca.reshape(X_ref, -1, 1)

    # Solver setup
    nlp = {'x': decision_vars, 'f': obj, 'g': g, 'p': params}
    opts = {"ipopt.print_level": 0, "print_time": 0}

    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    # Return solver and useful handles
    solver_vars = {
        'X': X,
        'U': U,
        'X_ref': X_ref,
        'N': N,
        'dt': dt,
        'n_states': 3,
        'n_controls': 2
    }

    return solver, solver_vars



## Variable initialization
sim_time = []


print('Program started')


client = RemoteAPIClient('localhost',23004)
sim = client.getObject('sim')

visionSensorHandle = sim.getObject('/Vision_sensor')
fl_w = sim.getObject('/flw')
fr_w = sim.getObject('/frw')
rr_w = sim.getObject('/rrw')
rl_w = sim.getObject('/rlw')
IMU = sim.getObject('/Accelerometer_forceSensor')
#COM = sim.getObject('/Husky/ReferenceFrame')
COM = sim.getObject('/Husky/Accelerometer/Accelerometer_mass')
Husky_ref = sim.getObject('/Husky')


defaultIdleFps = sim.getInt32Param(sim.intparam_idle_fps)
sim.setInt32Param(sim.intparam_idle_fps, 0)

# Run a simulation in stepping mode:
client.setStepping(True)
sim.startSimulation()

def process_img(img):
    img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
            # In CoppeliaSim images are left to right (x-axis), and bottom to top (y-axis)
            # (consistent with the axes of vision sensors, pointing Z outwards, Y up)
            # and color format is RGB triplets, whereas OpenCV uses BGR:
    img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0) #Convert to Grayscale

            # Current image
    cropped_image = img[288:480, 0:640] # Crop image to only to relevant path data (Done heuristically)
    bw_img = cv2.threshold(cropped_image, 125, 255, cv2.THRESH_BINARY)[1]  # convert the grayscale image to binary image
    return bw_img
    


while (t:= sim.getSimulationTime()) < 600:
    
    img, resX, resY = sim.getVisionSensorCharImage(visionSensorHandle)
    im_bw = process_img(img)

    H = Image2Waypoints.getHomography()
    scale = 100
    canvas_width = int(3.5 * scale)
    canvas_height = int(5.0 * scale)
    warped = cv2.warpPerspective(im_bw, H, (canvas_width, canvas_height))
    warped = cv2.bitwise_not(warped)
    height, width = warped.shape  # Or: warped.shape[:2]

    fitted_path, band_points = Image2Waypoints.fit_quadratic_on_raw_image(im_bw)

    waypoints_meters = Image2Waypoints.convert_pixel_path_to_waypoints_in_meters(
        pixel_path=fitted_path,
        warped_shape=(height, width),
        scale=100  # use the same scale as used in warp
    )


    # --- Step 2: Setup NMPC and Interpolate Waypoints ---
    solver, vars = create_nmpc_solver(N=10, dt=0.05)
    X_ref_np = interpolate_waypoints_to_horizon(waypoints_meters, N=vars['N'])
    X_ref_flat = X_ref_np.reshape((-1, 1))

    # Initial guess and state
    x0 = np.array([0.0, 0.0, 0.0])
    X_init = np.tile(x0.reshape(3, 1), (1, vars['N']+1))
    U_init = np.zeros((2, vars['N']))
    initial_guess = np.concatenate([X_init.reshape(-1, 1), U_init.reshape(-1, 1)], axis=0)

    # --- Step 3: Solve NMPC and Get Control ---
    sol = solver(x0=initial_guess, p=X_ref_flat, lbg=0, ubg=0)
    solution = sol['x'].full().flatten()
    U_opt = solution[3 * (vars['N']+1):].reshape((2, vars['N']))
    v_cmd, omega_cmd = U_opt[:, 0]

    print(f"Apply: v = {v_cmd:.3f}, omega = {omega_cmd:.3f}")


    #### set wheel velocities

    t_a = 0.0770 # Virtual Radius
    t_b = 0.0870 #Virtual Radius/ Virtual Trackwidth
        
    A = np.array([[t_a,t_a],[-t_b,t_b]])

    velocity = np.array([v_cmd,omega_cmd])
    phi_dots = np.matmul(inv(A),velocity) #Inverse Kinematics
    phi_dots = phi_dots.astype(float)
    Left = phi_dots[0].item()
    Right = phi_dots[1].item()

    sim.setJointTargetVelocity(fl_w, Left)
    sim.setJointTargetVelocity(fr_w, Right)
    sim.setJointTargetVelocity(rl_w, Left)
    sim.setJointTargetVelocity(rr_w, Right)   

    '''
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(waypoints_meters[:, 0], waypoints_meters[:, 1], 'bo-', label="Waypoints (m)")
    plt.xlabel("Lateral offset (m)")
    plt.ylabel("Forward distance (m)")
    plt.title("Waypoints in Robot Frame")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.show()
    '''
    

    sim_time.append(t)
    print(t)

    client.step()  # triggers next simulation step



sim.stopSimulation()