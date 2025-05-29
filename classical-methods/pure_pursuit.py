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

def find_lookahead_point(path, L):
    """
    Find the first point at least L meters ahead of the robot.
    Assumes path is in robot's body frame (x forward).
    """
    for pt in path:
        x, y = pt
        dist = np.sqrt(x**2 + y**2)
        if dist >= L:
            return pt
    return path[-1]  # fallback: last point if no one far enough


def compute_pure_pursuit_control(lookahead_pt, v, L):
    """
    Compute omega given a lookahead point and robot's desired speed.
    """
    k=0.5
    x, y = lookahead_pt
    alpha = np.arctan2(y, x)  # angle to target in robot frame
    omega = k*2 * v * np.sin(alpha) / L
    return v, omega



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

    lookahead_distance = 10 # meters
    v_cmd = 0.3  # constant speed

    lookahead_pt = find_lookahead_point(waypoints_meters, lookahead_distance)
    v_cmd, omega_cmd = compute_pure_pursuit_control(lookahead_pt, v_cmd, lookahead_distance)


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