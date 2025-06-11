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


def interpolate_waypoints_image_plane(waypoints, N):
    x_path = waypoints[:, 0]
    y_path = waypoints[:, 1]
    distances = np.insert(np.cumsum(np.linalg.norm(np.diff(waypoints, axis=0), axis=1)), 0, 0)
    total_length = distances[-1]

    fx = interp1d(distances, x_path, kind='linear', fill_value='extrapolate')
    fy = interp1d(distances, y_path, kind='linear', fill_value='extrapolate')

    interp_dists = np.linspace(0, total_length, N + 1)
    x_ref = fx(interp_dists)
    y_ref = fy(interp_dists)

    dx = np.gradient(x_ref)
    dy = np.gradient(y_ref)
    theta_ref = np.arctan2(dy, dx)

    return np.vstack([x_ref, y_ref, theta_ref])



def create_image_plane_mpc(N=10, dt=0.01, Q_weights=[0.0, 0.1, 0.1], R_weights=[0.1, 0.1], v_des=1000.0, alpha=0.1):
    x = ca.SX.sym('x')  # image x
    y = ca.SX.sym('y')  # image y
    theta = ca.SX.sym('theta')
    v = ca.SX.sym('v')
    omega = ca.SX.sym('omega')

    states = ca.vertcat(x, y, theta)
    controls = ca.vertcat(v, omega)

    rhs = ca.vertcat(
        v * ca.cos(theta),
        v * ca.sin(theta),
        omega
    )

    f = ca.Function('f', [states, controls], [rhs])

    X = ca.SX.sym('X', 3, N+1)
    U = ca.SX.sym('U', 2, N)
    X_ref = ca.SX.sym('X_ref', 3, N+1)

    Q = ca.diag(ca.SX(Q_weights))
    R = ca.diag(ca.SX(R_weights))

    obj = 0
    g = []
    g.append(X[:, 0] - X_ref[:, 0])

    for k in range(N):
        x_next = X[:, k] + dt * f(X[:, k], U[:, k])
        g.append(X[:, k+1] - x_next)

        obj += ca.mtimes([(X[:, k] - X_ref[:, k]).T, Q, (X[:, k] - X_ref[:, k])])
        obj += ca.mtimes([U[:, k].T, R, U[:, k]])
        obj += alpha * (U[0, k] - v_des) ** 2


    g = ca.vertcat(*g)
    decision_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
    params = ca.reshape(X_ref, -1, 1)

    nlp = {'x': decision_vars, 'f': obj, 'g': g, 'p': params}
    opts = {"ipopt.print_level": 0, "print_time": 0}

    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

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
COM = sim.getObject('/Husky/Accelerometer/Accelerometer_mass')
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
print(f'Reset values are{reset}')
print(f'shape of Reset is{len(reset)}')
if eval_log == True:
    log_vars = {
        'time' : [],
        'pose_X' : [],
        'pose_Y' : [],
        'linear_v' : [],
    }
else:
    reset = reset[0]
    log_vars = {
        'time' : [],
        'pose_X' : [],
        'pose_Y' : [],
        'linear_v' : [],
    }
    

for loc_counter, location in enumerate([reset]):
    print(f'location is {location}')
    
    #reset location (for a sequence of 10 location)
    sim.stopSimulation()
    while sim.getSimulationState() != sim.simulation_stopped:
        time.sleep(1)
    sim.setStepping(True)
    sim.startSimulation() 
    print(f'world handle is {sim.handle_world}')
    sim.setObjectPose(BodyFOR, location, sim.handle_world)
    client.step()
    time.sleep(1)
    client.step()


    sim_time = []

    last_valid_waypoints = None  # Store the most recent valid path

    while (t := sim.getSimulationTime()) < 30:
        
        img, resX, resY = sim.getVisionSensorCharImage(visionSensorHandle)
        im_bw = process_img(img)


        # Define the transform
        blur_transform = T.Compose([
            T.ToTensor(),
            T.GaussianBlur(kernel_size=3, sigma=0.001),  # fixed blur
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

        #H = Image2Waypoints.getHomography()
        #scale = 50
        #canvas_width = int(3.5 * scale)
        #canvas_height = int(5.0 * scale)
        #warped = cv2.warpPerspective(blurred_np, H, (canvas_width, canvas_height))
        #warped = cv2.bitwise_not(warped)
        #height, width = warped.shape

        fitted_path, band_points = Image2Waypoints.fit_quadratic_on_raw_image(blurred_np)

        img_height, img_width = 192, 640
        camera_center = np.array([img_width // 2, img_height - 1])

        # Interpolate path
        N = 25
        X_ref_img = interpolate_waypoints_image_plane(fitted_path, N)
        X_ref_flat = X_ref_img.reshape((-1, 1))

        # Initialize solver
        solver, vars = create_image_plane_mpc(N=N)
        x0 = np.array([camera_center[0], camera_center[1], -np.pi/2])  # Facing up
        X_init = np.tile(x0.reshape(3, 1), (1, N+1))
        U_init = np.zeros((2, N))
        initial_guess = np.concatenate([X_init.reshape(-1, 1), U_init.reshape(-1, 1)], axis=0)

        # Solve MPC
        sol = solver(x0=initial_guess, p=X_ref_flat, lbg=0, ubg=0)
        solution = sol['x'].full().flatten()
        U_opt = solution[3 * (vars['N']+1):].reshape((2, vars['N']))
        v_cmd, omega_cmd = U_opt[:, 20]
        v_cmd = np.clip(v_cmd,0,1.0)
        omega_cmd = np.clip(omega_cmd,-1.0,1.0)

        print(f"Apply: v = {v_cmd:.3f}, omega = {omega_cmd:.3f}")

        ########## visualization code #############
        # Convert MPC reference and fitted path to int for display
        ref_pts = np.vstack([X_ref_img[0], X_ref_img[1]]).T.astype(int)
        fitted_pts = fitted_path.astype(int)
        x_img, y_img = int(camera_center[0]), int(camera_center[1])
        theta_img = x0[2]  # current heading in image plane

        # Draw fitted curve (white)
        for pt in fitted_pts:
            cv2.circle(blurred_np, tuple(pt), radius=2, color=255, thickness=-1)

        # Draw MPC reference points (blue)
        for pt in ref_pts:
            cv2.circle(blurred_np, tuple(pt), radius=3, color=100, thickness=-1)

        # Draw camera center (red)
        cv2.circle(blurred_np, (x_img, y_img), radius=5, color=0, thickness=-1)

        # Draw heading line (green)
        line_len = 30  # in pixels
        tip_x = int(x_img + line_len * np.cos(theta_img))
        tip_y = int(y_img + line_len * np.sin(theta_img))
        cv2.line(blurred_np, (x_img, y_img), (tip_x, tip_y), color=128, thickness=2)

        # Show overlay
        cv2.imshow("Image Plane MPC Overlay", cv2.cvtColor(blurred_np, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)


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


    ######### save CSV of logged vairables ################
    #default save_path = None
    if save_path != None:
        import pandas as pd

        df_log = pd.DataFrame(log_vars)
        df_log.to_csv(f"{save_path}MPC_GB_100_{loc_counter}.csv", index=False)
        print(f"{loc_counter}_Log saved to csv")


sim.stopSimulation()   
print('Program ended')


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