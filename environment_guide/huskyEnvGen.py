

import time

import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy import savetxt
from numpy.linalg import inv
#from matplotlib.animation import FuncAnimation

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


print('Program started')



client = RemoteAPIClient('localhost',23004)
sim = client.getObject('sim')

'''
###### Section : Generate and save path from initial control points###########

ctrlPts = [0,0,0,0,0,0,1,1,0,0,0,0,0,1,1,1,0,0,0,0,1,0,1,0,0,0,0,1]

scale = 5

wp = [0,0,0,0,0,0,1, 2*scale,-2*scale,0,0,0,0,1, 4*scale,-1*scale,0,0,0,0,1, 6*scale,-2*scale,0,0,0,0,1, 8*scale,0,0,0,0,0,1, 6*scale,2*scale,0,0,0,0,1, 4*scale,1*scale,0,0,0,0,1, 2*scale,2*scale,0,0,0,0,1]
pathHandle = sim.createPath(wp, 2,100,1.0)

'''

pathHandle = sim.getObject('/PathL')

pathData = sim.unpackDoubleTable(sim.readCustomDataBlock(pathHandle, 'PATH'))

pathArray = np.array(pathData)
print(np.shape(pathData))


reshaped = np.reshape(pathArray,(250,7))
print(np.shape(reshaped))


#rev_pathL = [x_rev,y_rev]
#print(np.shape(rev_pathL))

#fig1 = plt.figure()
#plt.plot(reshaped[:,0], reshaped[:,1])
#plt.show()


import pandas as pd 
df = pd.DataFrame(reshaped)
#df.to_csv("path/to/dir/path_name.csv", header=False, index=False)




#fig2 = plt.figure()
#plt.plot(reshaped[:,3])
#plt.plot(reshaped[:,4])
#plt.plot(reshaped[:,5])
#plt.plot(reshaped[:,6])
#plt.show()


############### Section : Place Cones along the path points ##############

Cuboid = []
#primary_cone = sim.getObject('/Cone[0]')
#print(primary_cone)


#dummy = [None] * 100

for x in range(250):
        #print([cone_0])
        #cone_0 = sim.getObject(cone_0)
        #str_pt = '/Cone[' + str(int(x)) + ']' 
    #str_pt = '/Cone[0]' 
        #print(str_pt)
    Cuboid = sim.getObject('/Cuboid[0]')
    Cuboid = sim.copyPasteObjects([Cuboid],1)
    next_Cuboid = sim.getObject('/Cuboid[' + str(int(x+1)) + ']')
    #pose = sim.getObjectPose(cone0 , sim.handle_world)
        #pose[0] = reshaped[x,0]
        #pose[1] = reshaped[y,1]
        #cone_hand = sim.getObject('/Cone')
    sim.setObjectPose(next_Cuboid, [reshaped[x,0],reshaped[x,1],0.0254,0,0,0,1], sim.handle_world)
    del next_Cuboid
        #del pose






