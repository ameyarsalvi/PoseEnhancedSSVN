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
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from numpy import genfromtxt
from scipy.spatial.transform import Rotation
import torchvision.transforms as T
import os

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


####### Load Policy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load model with feature extractor
model = PPO.load("/home/asalvi/Downloads/bslnPEVN.zip", device= 'cuda')
## Variable initialization


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
Husky_ref = sim.getObject('/Husky')

defaultIdleFps = sim.getInt32Param(sim.intparam_idle_fps)
sim.setInt32Param(sim.intparam_idle_fps, 0)
# Run a simulation in stepping mode:
client.setStepping(True)
sim.startSimulation()

##################### Evaluation /logging check ###############
## via argparser, default = 0
reset = getReset()
print(f'Reset values are{reset}')
print(f'shape of Reset is{len(reset)}')
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

        save_dir = "/home/asalvi/raw_imgs/"
        os.makedirs(save_dir, exist_ok=True)

        frame_count = 0

        sim_time = []
        log_vars = {
                'time' : [],
                'pose_X' : [],
                'pose_Y' : [],
                'linear_v' : [],
            }
        while (t:= sim.getSimulationTime()) < 30:
            
            img, resX, resY = sim.getVisionSensorCharImage(visionSensorHandle)
            im_bw = process_img(img)

            im_bw_save = cv2.bitwise_not(im_bw)

            if im_bw is not None:
            # Save image
                filename = os.path.join(save_dir, f"frame_{frame_count:04d}.png")
                cv2.imwrite(filename, im_bw_save)
                print(f"Saved {filename}")
                frame_count += 1



            # Define the transform
            blur_transform = T.Compose([
                T.ToTensor(),
                T.GaussianBlur(kernel_size=15, sigma=100),  # fixed blur
                T.ToPILImage()
            ])
            
            # Apply blur
            blurred_img_pil = blur_transform(im_bw)

            # Convert back to NumPy (if needed)
            blurred_np = np.array(blurred_img_pil)


            # Show with OpenCV
            cv2.imshow("Augmented Image", cv2.cvtColor(blurred_np, cv2.COLOR_RGB2BGR))

            #resized = cv2.resize(im_bw, (320, 96), interpolation=cv2.INTER_NEAREST)
            #grayscale = np.expand_dims(resized, axis=-1).astype(np.uint8)  # (96, 320, 1)
            #transposed = np.transpose(grayscale, (2, 0, 1))  # (1, 96, 320)
            im_bw = cv2.resize(blurred_np, (0,0), fx=0.5, fy=0.5)
            im_bw = cv2.bitwise_not(im_bw)
            cv2.imshow('input image',im_bw)
            cv2.waitKey(1)
            im_bw_input = np.frombuffer(im_bw, dtype=np.uint8).reshape(96, 320,1)

            pred = model.policy.predict(im_bw_input)
            a = pred[0]
            V = 0.25*a[0].item() + 0.75
            OMG = 0.5*a[1].item()

            #### set wheel velocities

            t_a = 0.0770 # Virtual Radius
            t_b = 0.0870 #Virtual Radius/ Virtual Trackwidth
                
            A = np.array([[t_a,t_a],[-t_b,t_b]])

            #velocity = np.array([v_cmd,omega_cmd])
            velocity = np.array([V,OMG]) #Only for RL policy
            phi_dots = np.matmul(inv(A),velocity) #Inverse Kinematics
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

            client.step()  # triggers next simulation step

        ######### save CSV of logged vairables ################
        #default save_path = None
        if save_path != None:
            import pandas as pd

            df_log = pd.DataFrame(log_vars)
            df_log.to_csv(f"{save_path}RLPEVN_k_{kernal_val}_s_{sigma_val}_{loc_counter}.csv", index=False)
            print(f"{loc_counter}_Log saved to csv")


sim.stopSimulation()
