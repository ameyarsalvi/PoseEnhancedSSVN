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
from Image2Waypoints import Image2Waypoints
from Image2Waypoints2 import Image2Waypoints2
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from scipy.spatial.transform import Rotation
from numpy import genfromtxt
import random
import torchvision.transforms as T
import casadi as ca
from scipy.interpolate import interp1d
import torch
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image


#### Parse for logging arguments

import argparse
parser = argparse.ArgumentParser(description='Evaluation Logging Arguments')
parser.add_argument('--eval_log', type=bool, default= False, help='Should evaluation and logging be enabled')
parser.add_argument('--save_path', type=str, default= None, help='Directory to log evaluations')

args = parser.parse_args()
eval_log = args.eval_log
save_path = args.save_path

########## Controller specific functions

def getBodyVel(sim,COM):
    linear_vel, angular_vel = sim.getVelocity(COM)
    sRb = sim.getObjectMatrix(COM,sim.handle_world)
    Rot = np.array([[sRb[0],sRb[1],sRb[2]],[sRb[4],sRb[5],sRb[6]],[sRb[8],sRb[9],sRb[10]]])
    vel_body = np.matmul(np.transpose(Rot),np.array([[linear_vel[0]],[linear_vel[1]],[linear_vel[2]]]))
    realized_vel = np.abs(-1*vel_body[2].item())
    return realized_vel


def getReset():
    Reset = []
    path_loc = '/home/asalvi/code_workspace/Husky_CS_SB3/PoseEnhancedVN/train/MixPathFlip/'
    
    filenames = [
        'ArcPath1.csv', 'ArcPath2.csv', 'ArcPath3.csv', 'ArcPath4.csv', 'ArcPath5.csv',
        'ArcPath1_.csv', 'ArcPath2_.csv', 'ArcPath3_.csv', 'ArcPath4_.csv', 'ArcPath5_.csv'
    ]
    
    for fname in filenames:
        path = genfromtxt(path_loc + fname, delimiter=',')
        if path.ndim == 1:
            path = path[np.newaxis, :]  # handle single-line case

        yaw = path[0, 2]
        rot = Rotation.from_euler('xyz', [0, 0, yaw], degrees=False)
        rot_quat = rot.as_quat()  # [x, y, z, w]

        position = [path[0, 0], path[0, 1], 0.325,
                    rot_quat[0], rot_quat[1], rot_quat[2], rot_quat[3]]
        Reset.append(position)

    return Reset

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
import numpy as np

def create_nmpc_solver(N, dt,
                       Q_weights=[0.1, 0.1, 0.5],
                       R_weights=[0.1, 0.1],
                       v_des=0.75,
                       alpha=0.9,
                       v_bounds=(0.0, 1.0),
                       omega_bounds=(-1.0, 1.0)):
    """
    Creates a nonlinear MPC solver for a differential drive robot using CasADi,
    including soft tracking of a desired linear velocity and bounded controls.

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
        v_bounds : tuple (v_min, v_max)
            Bounds on linear velocity (m/s).
        omega_bounds : tuple (omega_min, omega_max)
            Bounds on angular velocity (rad/s).

    Returns:
        solver : CasADi solver object
        solver_vars : dict with symbolic variables and parameters
        lbx : lower bounds on decision variables
        ubx : upper bounds on decision variables
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

    g.append(X[:, 0] - X_ref[:, 0])  # initial condition

    for k in range(N):
        x_next = X[:, k] + dt * f(X[:, k], U[:, k])
        g.append(X[:, k+1] - x_next)

        # Path tracking cost
        obj += ca.mtimes([(X[:, k] - X_ref[:, k]).T, Q, (X[:, k] - X_ref[:, k])])
        obj += ca.mtimes([U[:, k].T, R, U[:, k]])

        # Soft velocity tracking penalty
        obj += alpha * (U[0, k] - v_des)**2

    g = ca.vertcat(*g)

    # Flatten decision variables and parameters
    decision_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
    params = ca.reshape(X_ref, -1, 1)

    # Define bounds
    n_X = X.size1() * (N + 1)  # 3 * (N+1)
    n_U = U.size1() * N        # 2 * N

    lbx = [-ca.inf] * n_X
    ubx = [ ca.inf] * n_X

    for _ in range(N):
        lbx += [v_bounds[0], omega_bounds[0]]
        ubx += [v_bounds[1], omega_bounds[1]]

    lbx = ca.vertcat(*lbx)
    ubx = ca.vertcat(*ubx)

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

    return solver, solver_vars, lbx, ubx



################### Standard variable initialization#############
print('Program started')


client = RemoteAPIClient('localhost',23004)
sim = client.getObject('sim')

visionSensorHandle = sim.getObject('/Vision_sensor')
fl_w = sim.getObject('/flw')
fr_w = sim.getObject('/frw')
rr_w = sim.getObject('/rrw')
rl_w = sim.getObject('/rlw')
IMU = sim.getObject('/Accelerometer_forceSensor')
COM = sim.getObject('/Husky')
Husky_ref = sim.getObject('/Husky')
BodyFOR = sim.getObject('/FORBody')
HuskyPos = sim.getObject('/FORBody/Husky/ReferenceFrame')
AbsFrame = sim.handle_world


defaultIdleFps = sim.getInt32Param(sim.intparam_idle_fps)
sim.setInt32Param(sim.intparam_idle_fps, 0)
# Run a simulation in stepping mode:
client.setStepping(True)
sim.startSimulation()
client.step()  

##################### Evaluation /logging check ###############
## via argparser, default = 0
reset = getReset()

if eval_log == True:
    blur = {
        'kernal' : [3, 15, 25, 35, 45],
        'sigma' : [0.001, 5, 15, 35, 55]
    }
else:
    reset = [reset[0]]
    blur = {
        'kernal' : [3],
        'sigma' : [0.001]
    }


for sigma_val, kernal_val in zip(blur['sigma'], blur['kernal']):  

    for loc_counter, location in enumerate(reset):
        print(f'location is {location}')
        
        #reset location (for a sequence of 10 location)
        sim.stopSimulation()
        while sim.getSimulationState() != sim.simulation_stopped:
            time.sleep(1)
        sim.setStepping(True)
        sim.startSimulation() 
        sim.setObjectPose(BodyFOR, location, sim.handle_world)
        client.step()
        time.sleep(1)
        client.step()


        sim_time = []
        log_vars = {
            'time' : [],
            'pose_X' : [],
            'pose_Y' : [],
            'linear_v' : [],
        }

        last_valid_waypoints = None  # Store the most recent valid path

        while (t := sim.getSimulationTime()) < 30:
            
            img, resX, resY = sim.getVisionSensorCharImage(visionSensorHandle)
            im_bw = process_img(img)

            # Define the transform
            blur_transform = T.Compose([
                T.ToTensor(),
                T.GaussianBlur(kernel_size=kernal_val, sigma=sigma_val),  # fixed blur
                T.ToPILImage()
            ])
            
            # Apply blur
            blurred_img_pil = blur_transform(im_bw)

            # Convert back to NumPy (if needed)
            blurred_np = np.array(blurred_img_pil)

            # Show with OpenCV
            cv2.imshow("Augmented Image", cv2.cvtColor(blurred_np, cv2.COLOR_RGB2BGR))


            #im_bw = randomize_pixel_location(im_bw,0.5)

            while im_bw.all() == None:
                time.sleep(0.01)
            cv2.imshow('bw image', im_bw)
            #cv2.waitKey(1)

            left_path, right_path, center_path, vis_img = Image2Waypoints2.fit_dual_quadratics_on_raw_image(im_bw)
            #print(f"length of waypoints is :{len(center_path)}")

            if vis_img is not None:
                cv2.imshow("Dual Lane Fit", vis_img)
                #cv2.waitKey(0)

            '''    
            H_img_to_world = np.array([
                [ 1.47455097e-02, -5.19979651e-03, -3.72375854e+00],
                [-3.78156614e-03, -2.76291572e-02,  1.90269435e+01],
                [-1.31813484e-03,  8.98766154e-03,  1.00000000e+00]
            ])
            '''
            H_img_to_world = np.array([
                [ 1.04660635e-02,  2.54905363e-03, -1.87768491e+00],
                [-3.84114181e-04, -1.30146589e-02,  4.17121481e+00],
                [-1.13487030e-03,  1.14988151e-02,  1.00000000e+00]
            ])

            if center_path is not None:
                waypoints = Image2Waypoints2.convert_pixel_path_to_waypoints(center_path, H_img_to_world)
                last_valid_waypoints = waypoints
            else:
                print("[MPC Loop] Warning: fitted_path is None. Reusing last valid waypoints.")

            #Image2Waypoints2.plot_waypoints(last_valid_waypoints)

            if last_valid_waypoints is None:
                print("[MPC Loop] No valid waypoints available. Skipping MPC step.")
                client.step()
                continue


            print(f"length of valid waypoints is :{len(last_valid_waypoints)}")      

            # --- Step 2: Setup NMPC and Interpolate Waypoints ---
            #solver, vars = create_nmpc_solver(N=10, dt=0.01)
            solver, vars, lbx, ubx = create_nmpc_solver(N=10, dt=0.02)

            X_ref_np = interpolate_waypoints_to_horizon(last_valid_waypoints, N=vars['N'])
            X_ref_flat = X_ref_np.reshape((-1, 1))

            x0 = np.array([0.0, 0.0, 0.0])
            X_init = np.tile(x0.reshape(3, 1), (1, vars['N']+1))
            U_init = np.zeros((2, vars['N']))
            initial_guess = np.concatenate([X_init.reshape(-1, 1), U_init.reshape(-1, 1)], axis=0)

            # --- Step 3: Solve NMPC and Get Control ---
            #sol = solver(x0=initial_guess, p=X_ref_flat, lbg=0, ubg=0)
            sol = solver(x0=initial_guess,p=X_ref_flat,lbx=lbx,ubx=ubx,lbg=0,ubg=0)
            solution = sol['x'].full().flatten()
            U_opt = solution[3 * (vars['N']+1):].reshape((2, vars['N']))
            v_cmd, omega_cmd = U_opt[:, 1]

            print(f"Apply: v = {v_cmd:.3f}, omega = {omega_cmd:.3f}")

            # --- Convert to wheel velocities ---
            t_a = 0.0770  # Virtual radius
            t_b = 0.0870  # Virtual trackwidth
            A = np.array([[t_a, t_a], [-t_b, t_b]])
            velocity = np.array([v_cmd, omega_cmd])
            phi_dots = np.matmul(inv(A), velocity)
            phi_dots = phi_dots.astype(float)
            Left = phi_dots[0].item()
            Right = phi_dots[1].item()

            sim.setJointTargetVelocity(fl_w, Left)
            sim.setJointTargetVelocity(fr_w, Right)
            sim.setJointTargetVelocity(rl_w, Left)
            sim.setJointTargetVelocity(rr_w, Right)

            # --- Logging ---
            linear_v = getBodyVel(sim, COM)
            pose = sim.getObjectPose(HuskyPos, sim.handle_world)
            
            log_vars['linear_v'].append(linear_v)
            log_vars['pose_X'].append(pose[0])
            log_vars['pose_Y'].append(pose[1])
            log_vars['time'].append(t)
            

            sim_time.append(t)
            print(t)

            client.step()
            cv2.waitKey(1)


        ######### save CSV of logged vairables ################
        #default save_path = None
        if save_path != None:
            import pandas as pd

            df_log = pd.DataFrame(log_vars)
            df_log.to_csv(f"{save_path}MPC_k_{kernal_val}_s_{sigma_val}_{loc_counter}.csv", index=False)
            print(f"{loc_counter}_Log saved to csv")


sim.stopSimulation()   
print('Program ended')