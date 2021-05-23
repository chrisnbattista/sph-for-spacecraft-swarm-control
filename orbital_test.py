# This file uses International Space Station (ISS) orbital data to test the performance of the Multi-Agent Coordination (MAC) system in a Python testbed.

######### IMPORTS ##########

# Import libraries
from datetime import time
from os import times
from matplotlib import colors
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
import scipy
import itertools
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from time import gmtime, strftime
import matplotlib.animation as animation
from multi_agent_kinetics import worlds, sim
from cosmos_wrapper import SPHController
from multi_agent_kinetics import forces, viz

######### SETUP SIM FROM DATA ##########

# Load ISS orbit data
data = pd.read_csv('./data/orbit_data2_ijk.csv', index_col=None)
EARTH_RAD = 6371000 # m

# Mission parameters
INTER_SAT_DELAY = 1 * 60 # 1 minute
TARGET_INTER_SAT_DELAY = INTER_SAT_DELAY * 1.05 # 10% larger separation - should be achievable in one orbit
STEPS_TO_RUN = 6000 # 556000 @ TS=0.01 = approx. one full orbit period. must be divisible by 10 due to progress bar (sim is done in 10 even steps) and 25 due to video
TIMESTEP = 0.1 # 100 Hz for both simulator physics and controller

# Video parameters
VIDEO_LENGTH = 10 # sec
FRAME_RATE = 25 # fps
STATES_PER_FRAME = int(STEPS_TO_RUN / (VIDEO_LENGTH * FRAME_RATE))

print(f'Steps to run: {STEPS_TO_RUN} @ timestep {TIMESTEP}\nInitial delay: {INTER_SAT_DELAY} sec\nTarget delay: {TARGET_INTER_SAT_DELAY} sec')

# Define delay function
def orbital_delay(iss_data, row, delay):
    '''Given a row in iss_data, find the row that corresponds to the given delay in seconds.'''
    delayed_i = int(row+delay) % iss_data.shape[0]
    return iss_data.iloc[delayed_i]

def iss_position(iss_data, delay):
    '''Returns a callback for the properly formatted position on ISS orbit for a particular delay, parameterized by time.'''
    def callback(time):
        d = orbital_delay(iss_data, int(time), delay)
        return np.array([
            d['I_position (m)'],
            d['J_position (m)'],
            d['K_position (m)']
        ])
    return callback

# Define constraint to map ISS acceleration into propagator?

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
    'h':10000,
    'h_attractor':50000,
    'Re':20,
    'a_max':0,
    'v_max':15,
    'M':1,
    'gamma':1,
    'rho_0':1,
    'inter_agent_w':1,
    'attractor_w':1,
    'obstacle_w':0,
    'x_target_pos':0,
    'y_target_pos':0,
    'z_target_pos':0
}
cosmos_compatible_mac_mission_params = {
    'sim_params':mac_mission_params
}

######### RUN SIM ##########

# Generate initial state by delaying ISS data
initial_state = sim.generate_generic_ic(
    [
        ( # id and time are auto-generated
            1, # mass
            orbital_delay(data, 0, -(4-i) * INTER_SAT_DELAY)['I_position (m)'], # delayed position X3
            orbital_delay(data, 0, -(4-i) * INTER_SAT_DELAY)['J_position (m)'],
            orbital_delay(data, 0, -(4-i) * INTER_SAT_DELAY)['K_position (m)'],
            orbital_delay(data, 0, -(4-i) * INTER_SAT_DELAY)['I_velocity (m/s)'], # delayed position's velocity X3
            orbital_delay(data, 0, -(4-i) * INTER_SAT_DELAY)['J_velocity (m/s)'],
            orbital_delay(data, 0, -(4-i) * INTER_SAT_DELAY)['K_velocity (m/s)']
        ) for i in range(5)
    ]
)
    
dists = scipy.spatial.distance.squareform(
    scipy.spatial.distance.pdist(initial_state[:,worlds.pos[3]])
)
initial_separations = np.array((dists[0,1], dists[1,2], dists[2,3], dists[3,4]))

# Initialize Python testbed simulator
sim_world = worlds.World(
    initial_state=initial_state,
    spatial_dims=3,
    n_agents=5,
    controllers=[
        SPHController(
            name=translation_table['id'][i],
            params=cosmos_compatible_mac_mission_params,
            translation_table=translation_table,
            attractor=iss_position(delay=-(4-i)*TARGET_INTER_SAT_DELAY, iss_data=data)
        ) for i in range(len(translation_table['id']))],
    forces=[forces.gravity],
    n_timesteps=STEPS_TO_RUN+1,
    timestep=TIMESTEP
)
for i in tqdm(range(10)):
    sim_world.advance_state(int(STEPS_TO_RUN/10))

######### PLOT RESULTS ##########

print("Saving video and plotting...")

# Set stylesheet
plt.style.use('dark_background')
# Set up orbit and mission overview figures
#orbit_fig, orbit_ax = viz.set_up_3d_plot()
##mission_fig, mission_ax = viz.set_up_3d_plot()
video_fig, video_ax = viz.set_up_3d_plot()
# Set up plots figure
plot_fig, plot_ax = viz.set_up_plot()

# Plot Earth sphere
#viz.plot_earth(orbit_ax)

# Visualize ISS orbit data
def plot_orbit(ax, x, y, z):
    ax.scatter(x, y, z, c='lightgreen', s=1, alpha=0.01)

#plot_orbit(orbit_ax, data['I_position (m)'], data['J_position (m)'], data['K_position (m)'])
#plot_orbit(mission_ax, data['I_position (m)'], data['J_position (m)'], data['K_position (m)'])

# Plot mission overview
# def plot_mission_overview(ax, agent_x, agent_y, agent_z, target_x, target_y, target_z):
#     '''Takes plot axis and numpy arrays for the xyz ECI coordinates of orbit positions, agent positions, and target positions to plot.'''
#     ax.scatter(data['I_position (m)'], data['J_position (m)'], data['K_position (m)'], c='lightgreen', s=1, alpha=0.01)
#     ax.scatter(agent_x, agent_y, agent_z, c='red', s=8, marker='^', alpha=1)
#     ax.scatter(target_x, target_y, target_z, c='blue', marker='x')

# initial_pos = initial_state[:,worlds.pos[3]]
# initital_target_pos = np.array(
#         [
#             iss_position(delay=i*TARGET_INTER_SAT_DELAY, iss_data=data)(0) for i in range(5)
#         ]
#     )
#plot_mission_overview(mission_ax,
#                            initial_pos[:,0], initial_pos[:,1], initial_pos[:,2],
#                            initital_target_pos[:,0], initital_target_pos[:,1], initital_target_pos[:,2])

# Visualize satellite swarm orbit data
traj = sim_world.get_history()[:, worlds.pos[3]]
satellite_pt_colors = \
            [
            np.array((1,0,0,1)),
            np.array((0.5, 0.5, 0, 1)),
            np.array((0, 0.5, 0.5, 1)),
            np.array((0.5, 0, 0.5, 1)),
            np.array((0.33, 0.33, 0.33, 1))
            ]
            #list(itertools.chain(*[(
        #) for i in np.geomspace(0.01, 1, int(traj.shape[0]/5))
    #]))

# Label the axes etc for video plot
video_ax.xaxis._axinfo["grid"].update({"linewidth":0})
video_ax.yaxis._axinfo["grid"].update({"linewidth":0})
video_ax.zaxis._axinfo["grid"].update({"linewidth":0})

def render_satellites(ax, timestep):
    start_index = timestep * 5
    stop_index = start_index + 5
    ax.clear()
    viz.plot_earth(ax)
    ax.scatter(traj[start_index:stop_index,0], traj[start_index:stop_index,1],traj[start_index:stop_index,2], c=satellite_pt_colors, s=8, marker='^')

def render_targets(ax, render_time):
    for i in range(5):
        target = iss_position(delay=-(4-i)*TARGET_INTER_SAT_DELAY, iss_data=data)(render_time)
        ax.scatter(target[0], target[1], target[2], color=satellite_pt_colors[i], marker='x')

# Visualize swarm orbit data as video and export
def animate(i):
    state_number = i * STATES_PER_FRAME
    render_satellites(video_ax, state_number)
    render_time = int(state_number * TIMESTEP)
    target_positions = np.array([iss_position(delay=-(4-i)*TARGET_INTER_SAT_DELAY, iss_data=data)(render_time) for i in range(5)])
    actual_positions = sim_world.get_history()[state_number:state_number+5,worlds.pos[3]]
    #print(target_positions - actual_positions)
    render_targets(video_ax, render_time)

# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=25, metadata=dict(artist='Chris Battista'), bitrate=1800)

ani = FuncAnimation(
    fig=video_fig,
    func=animate,
    frames=(VIDEO_LENGTH * FRAME_RATE),
    interval=(1000 / FRAME_RATE)
)
ani.save(f'sim_{strftime("%Y-%m-%d %H:%M:%S", gmtime())}.mp4', writer=writer)

# Calculate and plot inter-agent distances
separations = np.zeros((int(traj.shape[0]/5), 5))
for i in range(0, int(traj.shape[0]/5)):
    separations[i,0] = i * TIMESTEP
    dists = scipy.spatial.distance.squareform(
        scipy.spatial.distance.pdist(traj[i*5:(i+1)*5,:])
    )
    separations[i,1:] = (dists[0,1], dists[1,2], dists[2,3], dists[3,4])

##distances = distances[:, np.all(distances < (mac_mission_params['internode_distance']/0.5858), axis=0)]

# Plot reference lines for separations plot
# h
#plot_ax.hlines(mac_mission_params['h'], 0, STEPS_TO_RUN*TIMESTEP, linestyles='dotted')
#plot_ax.hlines(mac_mission_params['h_attractor'], 0, STEPS_TO_RUN*TIMESTEP, linestyles='dotted', colors='green')
# 2h
#plot_ax.hlines(mac_mission_params['h']*2, 0, STEPS_TO_RUN*TIMESTEP, linestyles='dashed')
#plot_ax.hlines(mac_mission_params['h_attractor']*2, 0, STEPS_TO_RUN*TIMESTEP, linestyles='dashed', colors='green')
# attractor's target separation
target_sep = np.linalg.norm(
    iss_position(data, 0)(0) - iss_position(data, TARGET_INTER_SAT_DELAY)(0)
)
plot_ax.hlines(target_sep, 0, STEPS_TO_RUN*TIMESTEP, linestyles='dashdot')

# Plot separations over time
for i in range(4):
    plot_ax.plot(separations[:,0], separations[:,i+1])

# Make simple, bare axis lines through space:
# xAxisLine = ((min(data['I_position (m)'])/10, max(data['I_position (m)'])/10), (0, 0), (0,0))
# orbit_ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'gray')
# yAxisLine = ((0, 0), (min(data['J_position (m)'])/10, max(data['J_position (m)'])/10), (0,0))
# orbit_ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'gray')
# zAxisLine = ((0, 0), (0,0), (min(data['K_position (m)'])/10, max(data['K_position (m)'])/10))
# orbit_ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'gray')
 
# Label the axes etc for orbit plot
# orbit_ax.set_box_aspect([1,1,1])
# orbit_ax.xaxis._axinfo["grid"].update({"linewidth":0})
# orbit_ax.yaxis._axinfo["grid"].update({"linewidth":0})
# orbit_ax.zaxis._axinfo["grid"].update({"linewidth":0})
# orbit_ax.set_xlabel("ECI î")
# orbit_ax.set_ylabel("ECI ĵ")
# orbit_ax.set_zlabel("ECI k̂")
# orbit_ax.set_title("Test of MAC following chain-of-pearls attractors on ISS orbit line")

# Label the axes etc for separations plot
plot_ax.set_xlabel("Time (sec)")
plot_ax.set_ylabel("Separation distance (m)")
plot_ax.set_title("Inter-sat separation distances over time")
print(f'Initial separations: {initial_separations}')
print(f'Final separations: {separations[-1,1:]}')
print(f'Magnitude of desired separation change: {np.linalg.norm(iss_position(delay=TARGET_INTER_SAT_DELAY, iss_data=data)(0) - iss_position(delay=INTER_SAT_DELAY, iss_data=data)(0))}')
print(f'Separation changes: {np.array(separations[-1,1:]) - initial_separations}')

# Label the axes etc for mission plot
# mission_ax.set_box_aspect([1,1,1])
# mission_ax.xaxis._axinfo["grid"].update({"linewidth":0})
# mission_ax.yaxis._axinfo["grid"].update({"linewidth":0})
# mission_ax.zaxis._axinfo["grid"].update({"linewidth":0})
# mission_ax.set_xlabel("ECI î")
# mission_ax.set_ylabel("ECI ĵ")
# mission_ax.set_zlabel("ECI k̂")
# mission_ax.set_title("Red triangles = initial positions\nBlue circles = target positions")

plt.show()