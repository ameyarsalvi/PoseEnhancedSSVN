import numpy as np
from numpy import linalg
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Box 
from gymnasium.spaces import Dict
import random 
import torch
import cv2
from numpy.linalg import inv
from numpy import savetxt
import pickle
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


import os

import sys
sys.path.insert(0, "/home/asalvi/code_workspace/Husky_CS_SB3/train/")

import time
import math
from coppeliasim_zmqremoteapi_client import RemoteAPIClient



class HuskyCPEnvPathFren(Env):
    def __init__(self,port,seed,track_vel,paths_dir,variant,log_option,eval_dir):

        #Initializing socket connection
        client = RemoteAPIClient('localhost',port)
        self.sim = client.require('sim')
        self.sim.setStepping(True)
        self.sim.startSimulation()
 

        self.specific = variant
        self.EvalPath = eval_dir
        

        self.seed = seed
        #For Training
        self.track_vel = np.random.uniform(0.5,0.99)
        #For Evaluation
        #self.track_vel = track_vel
        #Throttle randomization for training if necssary
        #self.throttle = random.randint(1, 20)
        self.throttle = 1

        self.flw_vel = 0
        self.cam_err_old = 0
        self.enc_len = 0
        self.cent_err = []

        self.Xerr = 0*0.075*random.uniform(0, 1) #Position Randomization noise if necessary
        self.Yerr = 0*0.01*random.uniform(0, 1) # Poistion Randomization noise if necessary
        self.ICRx = 0

        self.log_var = log_option

        #Get object handles from CoppeliaSim handles
        self.visionSensorHandle = self.sim.getObject('/Vision_sensor')
        self.fl_w = self.sim.getObject('/flw')
        self.fr_w = self.sim.getObject('/frw')
        self.rr_w = self.sim.getObject('/rrw')
        self.rl_w = self.sim.getObject('/rlw')
        self.IMU = self.sim.getObject('/Accelerometer_forceSensor')
        self.COM = self.sim.getObject('/Husky')
        self.floor_cen = self.sim.getObject('/Floor')
        self.BodyFOR = self.sim.getObject('/FORBody')
        self.HuskyPos = self.sim.getObject('/FORBody/Husky/ReferenceFrame')
        self.CameraJoint = self.sim.getObject('/FORBody/Husky/Camera_joint')


        # Limits and definitions on Observation and action space

        self.action_space = Box(low=np.array([[-1],[-1]]), high=np.array([[1],[1]]),dtype=np.float32)

        self.observation_space = Dict({
            "image": Box(low=0, high=255, shape=(96, 320, 1), dtype=np.uint8),  # Image observation
            "vector": Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)  # Additional vector of 5 values
        })
        
       
        # Initial decleration of variables

        # Some global initialization
        # Reset initializations
       
        self.obs = []
        self.img_div = 0
        self.path_err_buff = []
        self.pose_err_buff = []


        # Paths
        from numpy import genfromtxt
        #path_loc = '/home/asalvi/code_workspace/Husky_CS_SB3/PoseEnhancedVN/train/MixPathFlip/'
        path_loc = paths_dir
        self.path1 = genfromtxt(path_loc + 'ArcPath1.csv', delimiter=',')
        self.path2 = genfromtxt(path_loc + 'ArcPath2.csv', delimiter=',')
        self.path3 = genfromtxt(path_loc + 'ArcPath3.csv', delimiter=',')
        self.path4 = genfromtxt(path_loc + 'ArcPath4.csv', delimiter=',')
        self.path5 = genfromtxt(path_loc + 'ArcPath5.csv', delimiter=',')
        self.path1_ = genfromtxt(path_loc + 'ArcPath1_.csv', delimiter=',')
        self.path2_ = genfromtxt(path_loc + 'ArcPath2_.csv', delimiter=',')
        self.path3_ = genfromtxt(path_loc + 'ArcPath3_.csv', delimiter=',')
        self.path4_ = genfromtxt(path_loc + 'ArcPath4_.csv', delimiter=',')
        self.path5_ = genfromtxt(path_loc + 'ArcPath5_.csv', delimiter=',')
        
        
        #log variables
        #self.log_err_feat = []
        #self.log_err_vel = []
        #self.log_err_feat_norm = []
        #self.log_rel_vel_ang = []
        #self.log_err_omega = []
        #self.log_err_omega_norm = []
        
        # actions
        self.log_actV = []
        self.log_actW = []
        
        # tracking objectivs
        self.log_err_path_norm = []
        self.log_err_pose_norm = []
        self.log_err_vel_norm = []
        
        # realized values
        self.log_rel_vel_lin = []
        self.log_rel_poseX = []
        self.log_rel_poseY = []
        
        # reference values
        self.log_ref_vel_lin = []
        self.log_ref_poseX = []
        self.log_ref_poseY = []
        self.log_ref_kappa = []
        
        
        


    def step(self,action):
 

        
        # For end2end (without IK) comment this and uncomment proceeding commented block
        self.GenControl(action)
        
        
        #Joint Velocities similar to how velocities are set on actual robot
        self.sim.setJointTargetVelocity(self.fl_w, self.Left)
        self.sim.setJointTargetVelocity(self.fr_w, self.Right)
        self.sim.setJointTargetVelocity(self.rl_w, self.Left)
        self.sim.setJointTargetVelocity(self.rr_w, self.Right)
        

        '''
        Fl_w = 6*action[0]
        Fr_w = 6*action[1]
        Rl_w = 6*action[0]
        Rr_w = 6*action[1]
        
        #Joint Velocities similar to how velocities are set on actual robot
        self.sim.setJointTargetVelocity(self.fl_w, Fl_w.item())
        self.sim.setJointTargetVelocity(self.fr_w, Fr_w.item())
        self.sim.setJointTargetVelocity(self.rl_w, Rl_w.item())
        self.sim.setJointTargetVelocity(self.rr_w, Rr_w.item())
        '''

        
         # Simulate step (CoppeliaSim Command) (Progress on the taken action)
        self.sim.step()
     
        # Get simulation data
        # IMAGE PROCESSING CODE
        img, resX, resY = self.sim.getVisionSensorCharImage(self.visionSensorHandle)
        self.img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
        self.img_preprocess()

        
        #Encoder lenght for dropbout based training, but not used here
        if self.step_no == 0:
            self.img_obs = self.obs
            self.enc_len = 1
            self.arc_dT = 0
        else:
            if self.step_no % self.throttle == 0:
                self.img_obs = self.obs
                self.enc_len = 1
                self.arc_dT =  0
            else:
                self.enc_len = self.enc_len + self.arc_dT
                pass
        

        #print(self.enc_len)
        self.enc_len_div = 1*self.enc_len
        self.enc_len_div = np.clip(self.enc_len_div,1,65)
        img_obs = self.img_obs
        #img_obs = np.divide(img_obs,(self.enc_len_div))

        # Generate additional vector data (replace this with your actual data)
        additional_info = np.array([self.track_vel],dtype=np.float32)

        #self.state = np.array(img_obs,dtype = np.uint8) #Just input image to CNN network
        # Combine into a dictionary observation
        self.state = {
            "image": np.array(img_obs, dtype=np.uint8),
            "vector": additional_info
        }
     

        #self.state = np.array(img_obs,dtype = np.uint8) #Just input image to CNN network


        self.getReward(action)
        reward = self.rew
    
        

        # Check for reset conditions
        # Removing episode length termination from reset condition
        if self.episode_length ==0 or np.abs(self.cent_err)>150:
        #if self.episode_length ==0 or np.abs(self.path_track_err)>1 or np.abs(self.pose_track_err)>0.01 :
            done = True
        else:
            done = False

        if self.log_var ==1:
            self.Logger()
        else:
            pass


        # Update Global variables
        self.episode_length -= 1 #Update episode step counter
        self.step_no += 1  

        info ={}

        return self.state, reward, done, False, info

    def render(self):
        pass

    def reset(self, seed=None):
        super().reset(seed=self.seed)

        #print("Here")

        # Reset initialization variables
        self.episode_length = 5000
        self.step_no = 0
        self.z_ang = []
        self.ArcLen = 0
        self.obs = []
        self.act0_prev =0
        self.act1_prev =0
        self.refVel = []

        self.camActPrev = 0
        self.camErrPrev = 0
        self.cam_err_old = 0
        
        self.enc_len = 0
        self.arc_dT = 0
        self.path_err_buff = []
        self.pose_err_buff = []
        self.current_pose = self.sim.getObjectPose(self.HuskyPos, self.sim.handle_world)


        #self.intg_pth_err = []
        self.track_vel = np.random.uniform(0.1,0.9)
        
        mass_randomizer = np.random.randint(5, size=1)
        #print(mass_randomizer)
        mass = 76 + mass_randomizer.item()
        #self.sim.setObjectFloatParam(self.COM, self.sim.shapefloatparam_mass , mass)

        self.sim.stopSimulation()
        while self.sim.getSimulationState() != self.sim.simulation_stopped:
            time.sleep(0.1)
        self.sim.setStepping(True)
        self.sim.startSimulation() 

        # Randomize spawning location so that learning is a bit more generalized
        # Three objects kept at different locations
        
        rand_spawn = np.random.randint(1, 11, 1, dtype=int)

        #rand_spawn = 9
        if rand_spawn == 1:
            # Create a rotation object from Euler angles specifying axes of rotation
            rot = Rotation.from_euler('xyz', [0, 0,  self.path1[0,2]], degrees=False)
            rot_quat = rot.as_quat()
            pose = [self.path1[0,0],self.path1[0,1],0.325,rot_quat[0],rot_quat[1],rot_quat[2],rot_quat[3]]
            self.path_selector = 1
        
        if rand_spawn == 2:
            # Create a rotation object from Euler angles specifying axes of rotation
            rot = Rotation.from_euler('xyz', [0, 0,  self.path1_[0,2]], degrees=False)
            rot_quat = rot.as_quat()
            pose = [self.path1_[0,0],self.path1_[0,1],0.325,rot_quat[0],rot_quat[1],rot_quat[2],rot_quat[3]]
            self.path_selector = 2

        elif rand_spawn == 3:
            rot = Rotation.from_euler('xyz', [0, 0, self.path2[0,2]], degrees=False)
            rot_quat = rot.as_quat()
            pose = [self.path2[0,0],self.path2[0,1],0.325,rot_quat[0],rot_quat[1],rot_quat[2],rot_quat[3]]
            self.path_selector = 3

        if rand_spawn == 4:
            # Create a rotation object from Euler angles specifying axes of rotation
            rot = Rotation.from_euler('xyz', [0, 0,  self.path2_[0,2]], degrees=False)
            rot_quat = rot.as_quat()
            pose = [self.path2_[0,0],self.path2_[0,1],0.325,rot_quat[0],rot_quat[1],rot_quat[2],rot_quat[3]]
            self.path_selector = 4

        elif rand_spawn == 5:
            rot = Rotation.from_euler('xyz', [0, 0, self.path3[0,2]], degrees=False)
            rot_quat = rot.as_quat()
            pose = [self.path3[0,0],self.path3[0,1],0.325,rot_quat[0],rot_quat[1],rot_quat[2],rot_quat[3]]
            self.path_selector = 5

        elif rand_spawn == 6:
            rot = Rotation.from_euler('xyz', [0, 0, self.path3_[0,2]], degrees=False)
            rot_quat = rot.as_quat()
            pose = [self.path3_[0,0],self.path3_[0,1],0.325,rot_quat[0],rot_quat[1],rot_quat[2],rot_quat[3]]
            self.path_selector = 6

        elif rand_spawn == 7:
            rot = Rotation.from_euler('xyz', [0, 0, self.path4[0,2]], degrees=False)
            rot_quat = rot.as_quat()
            pose = [self.path4[0,0],self.path4[0,1],0.325,rot_quat[0],rot_quat[1],rot_quat[2],rot_quat[3]]
            self.path_selector = 7
        
        elif rand_spawn == 8:
            rot = Rotation.from_euler('xyz', [0, 0, self.path4_[0,2]], degrees=False)
            rot_quat = rot.as_quat()
            pose = [self.path4_[0,0],self.path4_[0,1],0.325,rot_quat[0],rot_quat[1],rot_quat[2],rot_quat[3]]
            self.path_selector = 8

        elif rand_spawn == 9:
            rot = Rotation.from_euler('xyz', [0, 0, self.path5[0,2]], degrees=False)
            rot_quat = rot.as_quat()
            pose = [self.path5[0,0],self.path5[0,1],0.325,rot_quat[0],rot_quat[1],rot_quat[2],rot_quat[3]]
            self.path_selector = 9

        elif rand_spawn == 10:
            rot = Rotation.from_euler('xyz', [0, 0, self.path5_[0,2]], degrees=False)
            rot_quat = rot.as_quat()
            pose = [self.path5_[0,0],self.path5_[0,1],0.325,rot_quat[0],rot_quat[1],rot_quat[2],rot_quat[3]]
            self.path_selector = 10
    
        
        #spawn = self.sim.getObject('/Spawn1')
                    
        #pose = self.sim.getObjectPose(spawn, self.sim.handle_world)
        self.sim.setObjectPose(self.BodyFOR, pose,self.sim.handle_world)
        self.prev_pose = self.sim.getObjectPose(self.HuskyPos, self.sim.handle_world)

        ####### Sligthly reshuffle cones ############
        
        for x in range(250):

            cone1 = self.sim.getObject('/Cone[' + str(int(x+1)) + ']')
            cone1_pose = self.sim.getObjectPose(cone1, self.sim.handle_world)
            x_dist = 0.1*np.random.random(size=None)
            y_dist = 0.1*np.random.random(size=None)
            self.sim.setObjectPose(cone1, [cone1_pose[0]+x_dist, cone1_pose[1]+y_dist,cone1_pose[2], cone1_pose[3],cone1_pose[4],cone1_pose[5],cone1_pose[6]], self.sim.handle_world)


            cone2 = self.sim.getObject('/Cone2[' + str(int(x+1)) + ']')
            cone2_pose = self.sim.getObjectPose(cone2, self.sim.handle_world)
            x_dist = 0.1*np.random.random(size=None)
            y_dist = 0.1*np.random.random(size=None)
            self.sim.setObjectPose(cone2, [cone2_pose[0]+x_dist,cone2_pose[1]+y_dist,cone2_pose[2],cone2_pose[3],cone2_pose[4],cone2_pose[5],cone2_pose[6]], self.sim.handle_world)
        
        
    
        # IMAGE PROCESSING CODE (Only to send obseravation)
        img, resX, resY = self.sim.getVisionSensorCharImage(self.visionSensorHandle)
        self.img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
        self.img_preprocess()
        img_obs = self.obs

        additional_info = np.array([self.track_vel],dtype=np.float32)

        #self.state = np.array(img_obs,dtype = np.uint8) #Just input image to CNN network
        # Combine into a dictionary observation
        self.state = {
            "image": np.array(img_obs, dtype=np.uint8),
            "vector": additional_info
        }
        
        
        info = {}


        return self.state, info
    
    def arc_length(self,action):
        
        self.current_pose = self.sim.getObjectPose(self.HuskyPos, self.sim.handle_world)
        #self.curretn_orn = np.array([self.current_pose[3],self.current_pose[4],self.current_pose[5],self.current_pose[6]]) #
        
        dt_arc = np.sqrt(np.square(self.current_pose[0]-self.prev_pose[0]) + np.square(self.current_pose[1]-self.prev_pose[1]))
        self.ArcLen = self.ArcLen + dt_arc
        self.arc_dT = dt_arc
        delay_factor = 0

        if self.path_selector ==1:
            icp = min(enumerate(self.path1[:,9]), key=lambda x: abs(x[1]-self.ArcLen))
            l_idx = icp[0]+delay_factor
            des_poseX = self.path1[l_idx,0]
            des_poseY = self.path1[l_idx,1]
            rot = Rotation.from_euler('xyz', [0, 0, self.path1[l_idx,2]], degrees=False)
            rot_quat = rot.as_quat()
            des_poseQ = np.array([[rot_quat[0],rot_quat[1],rot_quat[2],rot_quat[3]]])
            ref_kappa = self.path4[icp[0],7]
            #des_poseQ = np.array([self.path1[icp[0],6],self.path1[icp[0],5],self.path1[icp[0],4],self.path1[icp[0],3]])
        elif self.path_selector == 3:
            icp = min(enumerate(self.path2[:,9]), key=lambda x: abs(x[1]-self.ArcLen))
            l_idx = icp[0]+delay_factor
            des_poseX = self.path2[l_idx,0]
            des_poseY = self.path2[l_idx,1]
            rot = Rotation.from_euler('xyz', [0, 0, self.path2[l_idx,2]], degrees=False)
            rot_quat = rot.as_quat()
            des_poseQ = np.array([[rot_quat[0],rot_quat[1],rot_quat[2],rot_quat[3]]])
            ref_kappa = self.path4[icp[0],7]
            #des_poseQ = np.array([self.path2[icp[0],6],self.path2[icp[0],5],self.path2[icp[0],4],self.path2[icp[0],3]])
        elif self.path_selector == 5:
            icp = min(enumerate(self.path3[:,9]), key=lambda x: abs(x[1]-self.ArcLen))
            l_idx = icp[0]+delay_factor
            des_poseX = self.path3[l_idx,0]
            des_poseY = self.path3[l_idx,1]
            rot = Rotation.from_euler('xyz', [0, 0, self.path3[l_idx,2]], degrees=False)
            rot_quat = rot.as_quat()
            des_poseQ =np.array([[rot_quat[0],rot_quat[1],rot_quat[2],rot_quat[3]]])
            ref_kappa = self.path4[icp[0],7]
            #des_poseQ = np.array([self.path3[icp[0],6],self.path3[icp[0],5],self.path3[icp[0],4],self.path3[icp[0],3]])
        elif self.path_selector == 7:
            icp = min(enumerate(self.path4[:,9]), key=lambda x: abs(x[1]-self.ArcLen))
            l_idx = icp[0]+delay_factor
            des_poseX = self.path4[l_idx,0]
            des_poseY = self.path4[l_idx,1]
            rot = Rotation.from_euler('xyz', [0, 0, self.path4[l_idx,2]], degrees=False)
            rot_quat = rot.as_quat()
            des_poseQ = np.array([[rot_quat[0],rot_quat[1],rot_quat[2],rot_quat[3]]])
            ref_kappa = self.path4[icp[0],7]
            #print(self.log_ref_kappa )
            #des_poseQ = np.array([self.path4[icp[0],6],self.path4[icp[0],5],self.path4[icp[0],4],self.path4[icp[0],3]])
        elif self.path_selector == 9:
            icp = min(enumerate(self.path5[:,9]), key=lambda x: abs(x[1]-self.ArcLen))
            l_idx = icp[0]+delay_factor
            des_poseX = self.path5[l_idx,0]
            des_poseY = self.path5[l_idx,1]
            rot = Rotation.from_euler('xyz', [0, 0, self.path5[l_idx,2]], degrees=False)
            rot_quat = rot.as_quat()
            des_poseQ = np.array([[rot_quat[0],rot_quat[1],rot_quat[2],rot_quat[3]]])
            ref_kappa = self.path4[icp[0],7]
        #    des_poseQ = np.array([self.path5[icp[0],6],self.path5[icp[0],5],self.path5[icp[0],4],self.path5[icp[0],3]])
            

        if self.path_selector ==2:
            icp = min(enumerate(self.path1_[:,9]), key=lambda x: abs(x[1]-self.ArcLen))
            l_idx = icp[0]+delay_factor
            des_poseX = self.path1_[l_idx,0]
            des_poseY = self.path1_[l_idx,1]
            rot = Rotation.from_euler('xyz', [0, 0, self.path1_[l_idx,2]], degrees=False)
            rot_quat = rot.as_quat()
            des_poseQ = np.array([[rot_quat[0],rot_quat[1],rot_quat[2],rot_quat[3]]])
            ref_kappa = self.path4[icp[0],7]
            #des_poseQ = np.array([self.path1[icp[0],6],self.path1[icp[0],5],self.path1[icp[0],4],self.path1[icp[0],3]])
        elif self.path_selector == 4:
            icp = min(enumerate(self.path2_[:,9]), key=lambda x: abs(x[1]-self.ArcLen))
            l_idx = icp[0]+delay_factor
            des_poseX = self.path2_[l_idx,0]
            des_poseY = self.path2_[l_idx,1]
            rot = Rotation.from_euler('xyz', [0, 0, self.path2_[l_idx,2]], degrees=False)
            rot_quat = rot.as_quat()
            des_poseQ = np.array([[rot_quat[0],rot_quat[1],rot_quat[2],rot_quat[3]]])
            ref_kappa = self.path4[icp[0],7]
            #des_poseQ = np.array([self.path2[icp[0],6],self.path2[icp[0],5],self.path2[icp[0],4],self.path2[icp[0],3]])
        elif self.path_selector == 6:
            icp = min(enumerate(self.path3_[:,9]), key=lambda x: abs(x[1]-self.ArcLen))
            l_idx = icp[0]+delay_factor
            des_poseX = self.path3_[l_idx,0]
            des_poseY = self.path3_[l_idx,1]
            rot = Rotation.from_euler('xyz', [0, 0, self.path3_[l_idx,2]], degrees=False)
            rot_quat = rot.as_quat()
            des_poseQ =np.array([[rot_quat[0],rot_quat[1],rot_quat[2],rot_quat[3]]])
            ref_kappa = self.path4[icp[0],7]
            #des_poseQ = np.array([self.path3[icp[0],6],self.path3[icp[0],5],self.path3[icp[0],4],self.path3[icp[0],3]])
        elif self.path_selector == 8:
            icp = min(enumerate(self.path4_[:,9]), key=lambda x: abs(x[1]-self.ArcLen))
            l_idx = icp[0]+delay_factor
            des_poseX = self.path4_[l_idx,0]
            des_poseY = self.path4_[l_idx,1]
            rot = Rotation.from_euler('xyz', [0, 0, self.path4_[l_idx,2]], degrees=False)
            rot_quat = rot.as_quat()
            des_poseQ = np.array([[rot_quat[0],rot_quat[1],rot_quat[2],rot_quat[3]]])
            ref_kappa = self.path4[icp[0],7]
            #des_poseQ = np.array([self.path4[icp[0],6],self.path4[icp[0],5],self.path4[icp[0],4],self.path4[icp[0],3]])
        elif self.path_selector == 10:
            icp = min(enumerate(self.path5_[:,9]), key=lambda x: abs(x[1]-self.ArcLen))
            l_idx = icp[0]+delay_factor
            des_poseX = self.path5_[l_idx,0]
            des_poseY = self.path5_[l_idx,1]
            rot = Rotation.from_euler('xyz', [0, 0, self.path5_[l_idx,2]], degrees=False)
            rot_quat = rot.as_quat()
            des_poseQ = np.array([[rot_quat[0],rot_quat[1],rot_quat[2],rot_quat[3]]])
            ref_kappa = self.path4[icp[0],7]
        #    des_poseQ = np.array([self.path5[icp[0],6],self.path5[icp[0],5],self.path5[icp[0],4],self.path5[icp[0],3]])
        
        
        self.state_prog(action)

        self.log_ref_kappa.append(ref_kappa)
        self.log_rel_poseX.append(self.current_pose[0])
        self.log_rel_poseY.append(self.current_pose[1])
        self.log_ref_poseX.append(des_poseX)
        self.log_ref_poseY.append(des_poseY)

        #self.log_ref_kappa = ref_kappa
        #self.log_rel_poseX = self.current_pose[0]
        #self.log_rel_poseY = self.current_pose[1]
        #self.log_ref_poseX = des_poseX
        #self.log_ref_poseY = des_poseY

        self.path_track_err = np.sqrt(np.square(self.current_pose[0].item()-des_poseX) + np.square(self.current_pose[1].item()-des_poseY))
        #self.path_track_err = np.sqrt(np.square(self.current_pose[0]-des_poseX) + np.square(self.current_pose[1]-des_poseY))
        self.path_err_buff.append(self.path_track_err)
        #self.pose_track_err = np.arccos(self.curretn_orn@des_poseQ) #
        currentQ = np.array([[self.current_pose[3],self.current_pose[4],self.current_pose[5],self.current_pose[6]]])
        self.pose_track_err = 1- np.dot(des_poseQ,np.transpose(currentQ))
        self.pose_err_buff.append(self.pose_track_err)
        #self.intg_pth_err.append(self.path_track_err)

        self.current_pose = self.sim.getObjectPose(self.HuskyPos, self.sim.handle_world)
        self.prev_pose = self.current_pose
        
    def state_prog(self,action):

        self.t_a = 0.0770 # Virtual Radius
        self.t_b = 0.0870 #Virtual Radius/ Virtual Trackwidth
        A = np.array([[self.t_a,self.t_a],[-self.t_b,self.t_b]])
        #print(A.shape)
        dt = 0.05

        sRb = self.sim.getObjectMatrix(self.COM,self.sim.handle_world)
        Rot = np.array([[sRb[0],sRb[1]],[sRb[4],sRb[5]]])
        #print(Rot.shape)

        body_Vel = np.dot(A,np.array([[action[0].item()],[action[1].item()]]))
        #print(body_Vel.shape)
        
        
        update = np.array([[self.prev_pose[0] + self.Xerr],[self.prev_pose[1] + self.Yerr]]) #Add noise to positions based off GPS data
        #update = np.array([[self.prev_pose[0]],[self.prev_pose[1]]])
        #print(update)
        self.current_pose[0] = update[0]
        self.current_pose[1] = update[1]

        #test0 = self.sim.getObjectPose(self.HuskyPos, self.sim.handle_world)
        #print(test0)
        

    def GenControl(self,action):

        self.V_lin = 0.25*action[0] + 0.75
        self.log_actV .append(self.V_lin)

        self.Omg_ang = 0.5*action[1] 
        self.log_actW.append(self.Omg_ang)

        # No condition
        self.t_a = 0.0770 # Virtual Radius
        self.t_b = 0.0870 #Virtual Radius/ Virtual Trackwidth
        
        
        #A = np.array([[self.t_a*0.0825,self.t_a*0.0825],[-0.1486/self.t_b,0.1486/self.t_b]])

        A = np.array([[self.t_a,self.t_a],[-self.t_b,self.t_b]])

        velocity = np.array([self.V_lin,self.Omg_ang])
        phi_dots = np.matmul(inv(A),velocity) #Inverse Kinematics
        phi_dots = phi_dots.astype(float)
        self.Left = phi_dots[0].item()
        self.Right = phi_dots[1].item()

        #### Camera Angle ## No reconfigurable camera

        CamPosition = self.sim.getJointPosition(self.CameraJoint)
        #print(CamPosition)
        cam_error = 0*(160-self.cX) #<< Error made zero
        CamFeed = 0.01*cam_error + 0.005*(np.abs(cam_error-self.cam_err_old))
        camSet = np.clip(CamFeed*(3.14/180),-3.14/2,3.14/2)
        self.cam_err_old = cam_error
        #self.sim.setJointPosition(self.CameraJoint, CamPosition +  0) #<< Camera joint never change
        
        self.img_div = camSet.item()/(0.5*3.14)
        self.img_div = 1  #<< No change in image pixels

    def img_preprocess(self):

        
        # In CoppeliaSim images are left to right (x-axis), and bottom to top (y-axis)
        # (consistent with the axes of vision sensors, pointing Z outwards, Y up)
        # and color format is RGB triplets, whereas OpenCV uses BGR:
        img0 = cv2.flip(cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY), 0) #Convert to Grayscale
        cropped_image = img0[288:480, 0:640] # Crop image to only to relevant path data (Done heuristically)
        cropped_image = cv2.resize(cropped_image, (0,0), fx=0.5, fy=0.5)
        im_bw = cv2.threshold(cropped_image, 125, 255, cv2.THRESH_BINARY)[1]  # convert the grayscale image to binary image
        #im_bw = cv2.threshold(im_bw, 125, 255, cv2.THRESH_BINARY)[1]  # convert the grayscale image to binary image
        noise = np.random.normal(0, 25, im_bw.shape).astype(np.uint8)
        noisy_image = cv2.add(im_bw, noise)
        im_bw = np.frombuffer(noisy_image, dtype=np.uint8).reshape(96, 320,1) # Reshape to required observation size
        self.obs = ~im_bw
        #cv2.imshow("obs", self.obs)
        #cv2.waitKey(1)

        
        cropped_image = img0[300:450, 0:640] # Crop image to only to relevant path data (Done heuristically)
        cropped_image = cv2.resize(cropped_image, (0,0), fx=0.5, fy=0.5)
        im_bw = cv2.threshold(cropped_image, 225, 250, cv2.THRESH_BINARY)[1]  # convert the grayscale image to binary image
        k = ~im_bw
        M = cv2.moments(k)
        #cv2.imshow("tape", k)
        #cv2.waitKey(1)
 
        # calculate x,y coordinate of center
        if M["m00"] != 0:
            self.cX = int(M["m10"] / M["m00"])
            self.cY = int(M["m01"] / M["m00"])
        else:
            self.cX, self.cY = 0, 0
    
    def lane_center(self):
        img0 = cv2.flip(cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY), 0) #Convert to Grayscale
        crop_error = img0[352:416, 0:640] # Crop image to only to relevant path data (Done heuristically)
        crop_error = cv2.threshold(crop_error, 225, 255, cv2.THRESH_BINARY)[1]  # convert the grayscale image to binary image
        crop_error = ~crop_error
        #cv2.imshow("For Calc", crop_error)
        #cv2.imshow("obs", im_bw_obs)
        #cv2.waitKey(1)

        M = cv2.moments(crop_error) # calculate moments of binary image
        # calculate x,y coordinate of center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        
        self.cX = cX
        
    

        # Centering Error :
        #self.error = 128 - cX #Lane centering error that can be used in reward function
    

    def getReward(self,action):

     # Calculate Reward
        '''
        This reward has three terms :
        1. Increase episode count (This just encourages to stay on the track)
        2. Increase linear velocity (This encourages to move so that the robot doesn't learn a trivial action)
        3. Reduce center tracing error (This encourages smoothening)
        '''

        # Linear velocity reward params

        linear_vel, angular_vel = self.sim.getVelocity(self.COM)
        sRb = self.sim.getObjectMatrix(self.COM,self.sim.handle_world)
        Rot = np.array([[sRb[0],sRb[1],sRb[2]],[sRb[4],sRb[5],sRb[6]],[sRb[8],sRb[9],sRb[10]]])
        vel_body = np.matmul(np.transpose(Rot),np.array([[linear_vel[0]],[linear_vel[1]],[linear_vel[2]]]))
        realized_vel = np.abs(-1*vel_body[2].item())
        self.log_rel_vel_lin.append(realized_vel)
        #disturb = np.clip(-0.25,0.0,0.25*np.sin(self.step_no))
        track_vel = self.track_vel
        self.log_ref_vel_lin.append(track_vel)
        err_vel = np.abs(track_vel - realized_vel)
        err_vel = np.clip(err_vel,0,1.0)
        norm_err_vel = (err_vel - 0)/(1.0) ##   << -------------- Normalized Linear Vel
        self.log_err_vel_norm.append(norm_err_vel)

        
        # Lane centering reward params
        img, resX, resY = self.sim.getVisionSensorCharImage(self.visionSensorHandle)
        self.img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
        self.lane_center()
        err_track = np.abs(self.cX - 160)
        if err_track > 160:
            err_track = 160   
        else:
            pass
        norm_err_track = (err_track - 0)/160 ##   << -------------- Normalized Lane centering
        self.cent_err = err_track
        print(self.cent_err)
        
        '''
        # Preview Maximizing error

        pix_sum = np.sum(self.obs)
        pix_sum = pix_sum/(255*96*320)
        '''

        # Angular velocity reward params
        Gyro_Z = self.sim.getFloatSignal("myGyroData_angZ")
        if Gyro_Z:
            err_effort = -1*np.abs(Gyro_Z) 
        else:
            err_effort = np.abs(0)     
        norm_err_eff = (np.abs(self.Omg_ang))/0.5 ##   << -------------- Normalized angular velocity

        # Path Tracking Reward Parameters
        self.arc_length(action) #returns pose tracking aswell
        path_track_arr = self.path_err_buff[-1:]
        err_pth = np.clip(path_track_arr,0,1)
        norm_err_path = (err_pth)/1
        self.log_err_path_norm.append(norm_err_path)


        # Pose Tracking Reward Parameters
        pose_track_arr = self.pose_err_buff[-1:]
        err_pose = np.clip(pose_track_arr,0,1)
        norm_err_pose = (err_pose)/1
        self.log_err_pose_norm.append(norm_err_pose)

        
        #Smoot Actuation effort
        act0_eff = action[0] - self.act0_prev
        norm_act0_eff = abs(act0_eff)/2
        
        act1_eff = action[0] - self.act1_prev
        norm_act1_eff = abs(act1_eff)/2
        
        self.act0_prev =action[0]
        self.act1_prev =action[1]


        #Reward Calculations
        self.rew =(1-norm_act0_eff)**2 + (1-norm_act1_eff)**2 + (1 - norm_err_vel)**2 + (1 - norm_err_track)**2
        self.rew = np.float64(self.rew.item())


    def Logger(self):

        specifier = self.specific

        import os 
        #EvalPath = '/home/asalvi/code_workspace/Husky_CS_SB3/Evaluation/Policies/expTrt/expDump/expDumpAll/'
        os.makedirs(self.EvalPath, exist_ok=True)
        
        
        #ACTIONS

        
        with open(self.EvalPath + specifier + "_actV", "wb") as fp:   #Pickling
            pickle.dump(self.log_actV, fp)
        
        
        with open(self.EvalPath + specifier + "_actW", "wb") as fp:   #Pickling
            pickle.dump(self.log_actW, fp)  
        
        # TRACKING OBJECTIVES
        
        
        with open(self.EvalPath + specifier + "_err_path_norm", "wb") as fp:   #Pickling
            pickle.dump(self.log_err_path_norm, fp)
            
        with open(self.EvalPath + specifier + "_err_pose_norm", "wb") as fp:   #Pickling
            pickle.dump(self.log_err_pose_norm, fp)
           
        
        with open(self.EvalPath + specifier + "_err_vel_norm", "wb") as fp:   #Pickling
            pickle.dump(self.log_err_vel_norm, fp)
            
        # REALIZED VALUES
            
        
        with open(self.EvalPath + specifier + "_rel_vel_lin", "wb") as fp:   #Pickling
            pickle.dump(self.log_rel_vel_lin, fp)
                
        
        with open(self.EvalPath + specifier + "_rel_poseX", "wb") as fp:   #Pickling
            pickle.dump(self.log_rel_poseX, fp)
            
        
        with open(self.EvalPath + specifier + "_rel_poseY", "wb") as fp:   #Pickling
            pickle.dump(self.log_rel_poseY, fp)
            
            
        # REFERENCE VALUES
        
        with open(self.EvalPath + specifier + "_ref_vel_lin", "wb") as fp:   #Pickling
            pickle.dump(self.log_ref_vel_lin, fp)
                
        with open(self.EvalPath + specifier + "_ref_poseX", "wb") as fp:   #Pickling
            pickle.dump(self.log_ref_poseX, fp)
 
        with open(self.EvalPath + specifier + "_ref_poseY", "wb") as fp:   #Pickling
            pickle.dump(self.log_ref_poseY, fp)

        with open(self.EvalPath + specifier + "_ref_kappa", "wb") as fp:   #Pickling
            pickle.dump(self.log_ref_kappa, fp)
        
        
        
        '''
        self.log_err_feat.append(err_track)
        with open(path + specifier + "_err_feat", "wb") as fp:   #Pickling
            pickle.dump(self.log_err_feat, fp)

        self.log_err_vel.append(err_vel)
        with open(path + specifier + "_err_vel", "wb") as fp:   #Pickling
            pickle.dump(self.log_err_vel, fp)

        self.log_err_omega.append(err_effort)
        with open(path + specifier + "_err_omega", "wb") as fp:   #Pickling
            pickle.dump(self.log_err_vel, fp)

        self.log_err_omega_norm.append(norm_err_eff)
        with open(path + specifier + "_err_omega_norm", "wb") as fp:   #Pickling
            pickle.dump(self.log_err_vel_norm, fp)
        
        
        self.log_rel_vel_ang.append(Gyro_Z)
        with open(path + specifier + "_rel_vel_ang", "wb") as fp:   #Pickling
            pickle.dump(self.log_rel_vel_ang, fp)
        '''



##### Environment Checker
'''
from stable_baselines3.common.env_checker import check_env

env = HuskyCPEnvPathFren(
                       port = 23004,
                       seed = 0, 
                       track_vel = 0.75, 
                       paths_dir = "/home/asalvi/code_workspace/Husky_CS_SB3/PoseEnhancedVN/train/MixPathFlip/", 
                       variant = "test", 
                       log_option = 0, 
                       eval_dir = "/home/asalvi/code_workspace/Husky_CS_SB3/Evaluation/Policies/expTrt/expDump/expDumpAll/")
# It will check your custom environment and output additional warnings if needed
check_env(env)
'''