# This file uses International Space Station (ISS) orbital data to test the performance of the Multi-Agent Coordination (MAC) system in a Python testbed.

# Import libraries
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from multi_agent_kinetics import worlds
from cosmos_wrapper import SPHController

# Load ISS orbit data
data = pd.read_csv('./uhm_hcl/data/orbit_data2_ijk.csv', index_col=None)
EARTH_RAD = 6371000

# Initialize foreign key translations between iCOSMOS and MAK
translation_table = {
    'id':{
        0:'mothership',
        1:'daughter_01',
        2:'daughter_02',
        3:'daughter_03',
        4:'daughter_04'
    }
}

# Initialize the mission parameters for MAC
mac_mission_params = {
    'internode_distance':5000,
    'Re':20,
    'a_max':0.03,
    'v_max':15,
    'M':1,
    'gamma':1,
    'rho_0':1,
    'inter_agent_w':1,
    'attractor_w':1,
    'obstacle_w':1,
    'x_target_pos':0,
    'y_target_pos':0,
    'z_target_pos':0
    }

# Initialize Python testbed simulator
sim_world = worlds.World(
    spatial_dims=3,
    n_agents=5,
    controllers=[
        SPHController(
            name=translation_table['id'][i],
            params=mac_mission_params,
            translation_table=translation_table
        ) for i in range(len(translation_table['id']))],
    n_timesteps=10000,
    timestep=10
)

# Set up figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(30, 185)

# plot Earth sphere
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v) * EARTH_RAD
y = np.sin(u)*np.sin(v) * EARTH_RAD
z = np.cos(v) * EARTH_RAD
ax.plot_wireframe(x, y, z, colors="gray", linewidths=0.5, alpha=0.4)

# Visualize ISS orbit data
ax.scatter(data['I_position (m)'], data['J_position (m)'], data['K_position (m)'], c='g', s=1, alpha=0.7)

# make simple, bare axis lines through space:
xAxisLine = ((min(data['I_position (m)'])/10, max(data['I_position (m)'])/10), (0, 0), (0,0))
ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'gray')
yAxisLine = ((0, 0), (min(data['J_position (m)'])/10, max(data['J_position (m)'])/10), (0,0))
ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'gray')
zAxisLine = ((0, 0), (0,0), (min(data['K_position (m)'])/10, max(data['K_position (m)'])/10))
ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'gray')
 
# label the axes
ax.xaxis._axinfo["grid"].update({"linewidth":0})
ax.yaxis._axinfo["grid"].update({"linewidth":0})
ax.zaxis._axinfo["grid"].update({"linewidth":0})
ax.set_xlabel("ECI î")
ax.set_ylabel("ECI ĵ")
ax.set_zlabel("ECI k̂")
ax.set_title("Test of MAC following chain-of-pearls attractors on ISS orbit line")

plt.show()