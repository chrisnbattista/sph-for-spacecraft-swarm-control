# This file uses International Space Station (ISS) orbital data to test the performance of the Multi-Agent Coordination (MAC) system in a Python testbed.

# Import libraries
from matplotlib import colors
import numpy as np
import pandas as pd
import scipy
import itertools
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from multi_agent_kinetics import worlds, sim
from cosmos_wrapper import SPHController
from multi_agent_kinetics import forces

# Load ISS orbit data
data = pd.read_csv('./uhm_hcl/data/orbit_data2_ijk.csv', index_col=None)
EARTH_RAD = 6371000
INTER_SAT_DELAY = 1 * 60 # 1 minute
TARGET_INTER_SAT_DELAY = INTER_SAT_DELAY * 1.1

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
    'h':200000,
    'h_attractor':200000,
    'Re':20,
    'a_max':0.03,
    'v_max':15,
    'M':1,
    'gamma':1,
    'rho_0':1,
    'inter_agent_w':1,
    'attractor_w':1e18,
    'obstacle_w':0,
    'x_target_pos':0,
    'y_target_pos':0,
    'z_target_pos':0
}
cosmos_compatible_mac_mission_params = {
    'sim_params':mac_mission_params
}

# Generate initial state by delaying ISS data
initial_state = sim.generate_generic_ic(
    [
        ( # id and time are auto-generated
            1, # mass
            orbital_delay(data, 0, i * INTER_SAT_DELAY)['I_position (m)'], # delayed position X3
            orbital_delay(data, 0, i * INTER_SAT_DELAY)['J_position (m)'],
            orbital_delay(data, 0, i * INTER_SAT_DELAY)['K_position (m)'],
            orbital_delay(data, 0, i * INTER_SAT_DELAY)['I_velocity (m/s)'], # delayed position's velocity X3
            orbital_delay(data, 0, i * INTER_SAT_DELAY)['J_velocity (m/s)'],
            orbital_delay(data, 0, i * INTER_SAT_DELAY)['K_velocity (m/s)']
        ) for i in range(5)
    ]
)

# Initialize Python testbed simulator
STEPS_TO_RUN = 40000
sim_world = worlds.World(
    initial_state=initial_state,
    spatial_dims=3,
    n_agents=5,
    controllers=[
        SPHController(
            name=translation_table['id'][i],
            params=cosmos_compatible_mac_mission_params,
            translation_table=translation_table,
            attractor=iss_position(delay=i*TARGET_INTER_SAT_DELAY, iss_data=data)
        ) for i in range(len(translation_table['id']))],
    forces=[forces.gravity],
    n_timesteps=STEPS_TO_RUN+1,
    timestep=0.1
)
for i in tqdm(range(10)):
    sim_world.advance_state(int(STEPS_TO_RUN/10))

# Set up orbit figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(30, 185)

# Set up plots figure
plot_fig = plt.figure()
plot_ax = plot_fig.add_subplot(111)
##alt_ax = plot_fig.add_subplot(211)

# Set up mission overview figure
mission_fig = plt.figure()
mission_ax = mission_fig.add_subplot(111, projection='3d')

# Plot mission overview
mission_ax.scatter(data['I_position (m)'], data['J_position (m)'], data['K_position (m)'], c='g', s=1, alpha=0.01)
initial_pos = initial_state[:,worlds.pos[3]]
mission_ax.scatter(initial_pos[:,0], initial_pos[:,1], initial_pos[:,2], c='red', s=8, marker='^', alpha=1)
initital_target_pos = np.array(
    [
        iss_position(delay=i*TARGET_INTER_SAT_DELAY, iss_data=data)(0) for i in range(5)
    ]
)
mission_ax.scatter(initital_target_pos[:,0], initital_target_pos[:,1], initital_target_pos[:,2], c='blue', marker='x')

# Plot Earth sphere
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v) * EARTH_RAD
y = np.sin(u)*np.sin(v) * EARTH_RAD
z = np.cos(v) * EARTH_RAD
ax.plot_wireframe(x, y, z, colors="gray", linewidths=0.5, alpha=0.4)

# Visualize ISS orbit data
ax.scatter(data['I_position (m)'], data['J_position (m)'], data['K_position (m)'], c='g', s=1, alpha=0.01)

# Visualize satellite swarm orbit data
traj = sim_world.get_history()[:, worlds.pos[3]][::21]
satellite_pt_colors = list(itertools.chain(*[
        (
            np.array((1,0,0,i)),
            np.array((0.5, 0.5, 0, i)),
            np.array((0, 0.5, 0.5, i)),
            np.array((0.5, 0, 0.5, i)),
            np.array((0.33, 0.33, 0.33, i))
        ) for i in np.geomspace(0.00001, 1, int(traj.shape[0]/5))
    ]))
ax.scatter(traj[:,0], traj[:,1],traj[:,2], c=satellite_pt_colors, s=8, marker='^')

# Calculate and plot inter-agent distances
separations = np.sort(
    np.array(
        [
            scipy.spatial.distance.pdist(obs) for obs in traj.reshape(-1, 5, traj.shape[1])
        ]
    )
)
shortest_separations = separations[:,:4]
##distances = distances[:, np.all(distances < (mac_mission_params['internode_distance']/0.5858), axis=0)]

# Plot reference distance lines

# h
plot_ax.hlines(mac_mission_params['h'], 0, STEPS_TO_RUN, linestyles='dotted')
plot_ax.hlines(mac_mission_params['h_attractor'], 0, STEPS_TO_RUN, linestyles='dotted', colors='green')
# 2h
plot_ax.hlines(mac_mission_params['h']*2, 0, STEPS_TO_RUN, linestyles='dashed')
plot_ax.hlines(mac_mission_params['h_attractor']*2, 0, STEPS_TO_RUN, linestyles='dashed', colors='green')
# attractor's target separation
target_sep = np.linalg.norm(
    iss_position(data, 0)(0) - iss_position(data, TARGET_INTER_SAT_DELAY)(0)
)
plot_ax.hlines(target_sep, 0, STEPS_TO_RUN, linestyles='dashdot')
for i in range(shortest_separations.shape[1]):
    plot_ax.plot(shortest_separations[:,i])

# Make simple, bare axis lines through space:
xAxisLine = ((min(data['I_position (m)'])/10, max(data['I_position (m)'])/10), (0, 0), (0,0))
ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'gray')
yAxisLine = ((0, 0), (min(data['J_position (m)'])/10, max(data['J_position (m)'])/10), (0,0))
ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'gray')
zAxisLine = ((0, 0), (0,0), (min(data['K_position (m)'])/10, max(data['K_position (m)'])/10))
ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'gray')
 
# Label the axes etc for orbit plot
ax.set_box_aspect([1,1,1])
ax.xaxis._axinfo["grid"].update({"linewidth":0})
ax.yaxis._axinfo["grid"].update({"linewidth":0})
ax.zaxis._axinfo["grid"].update({"linewidth":0})
ax.set_xlabel("ECI î")
ax.set_ylabel("ECI ĵ")
ax.set_zlabel("ECI k̂")
ax.set_title("Test of MAC following chain-of-pearls attractors on ISS orbit line")

# Label the axes etc for separations plot
plot_ax.set_xlabel("Time (sec)")
plot_ax.set_ylabel("Separation distance (m)")
plot_ax.set_title("4 shortest separation distances over time\nDashed=2h\nDotted=h")
print(f'Final separations: {separations}')

# Label the axes etc for mission plot
mission_ax.set_box_aspect([1,1,1])
mission_ax.xaxis._axinfo["grid"].update({"linewidth":0})
mission_ax.yaxis._axinfo["grid"].update({"linewidth":0})
mission_ax.zaxis._axinfo["grid"].update({"linewidth":0})
mission_ax.set_xlabel("ECI î")
mission_ax.set_ylabel("ECI ĵ")
mission_ax.set_zlabel("ECI k̂")
mission_ax.set_title("Red triangles = initial positions\nBlue circles = target positions")

plt.show()