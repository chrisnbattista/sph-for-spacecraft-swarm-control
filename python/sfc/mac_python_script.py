import json, math
import numpy as np
from sklearn.preprocessing import normalize

# Called from nodes being simulated
# Accepts as arguments:
#	a string for the name of the node
#	a json string of vector<sim_state>, which the python json module can automatically parse as a list of dicts
#	a json string of vector<sim_param>, which the python json module can automatically parse as a list of dicts

#	1) Read in the state vectors and parameters from cosmosstruc
#	2) Parse the data structure and modify the values in the python script
#	3) Return back the modified state vectors to the agent

# Define kernel scaling factors
def quadratic_scaling_factor_3d(h):
	"""Returns the correct scaling factor for the 3D version of the quadratic kernel."""
	return 3.0/(16*math.pi) * h**3

def spline_scaling_factor_3d(h):
	"""Returns the correct scaling factor for the 3D version of the B-spline kernel."""
	return 1.0 / (6*math.pi**2) * h**3

# Define the kernel functions
# TO DO: insert B-spline kernel once derivation complete for 3D
def quadratic(r, h):
	"""Implements a quadratic kernel function with support radius of 2h."""
	if r/h < 2:
		return 	quadratic_scaling_factor_3d(h) \
				* 15.0 / (16 * math.pi * h**3) \
				* ((r/h)**2 / 4 - r/h + 1)
	else:
		return 0

def quadratic_grad(r, h):
	"""Implements the gradient of a quadratic kernel function with support radius of 2h."""
	if r/h < 2:
		return 	quadratic_scaling_factor_3d(h) \
				* 15.0 / (16 * math.pi * h**4) \
				* (r/h/2 - 1)
	else:
		return 0

# Define the property function
def pairwise_density(c_squared, rho_0, r_ij, h):
	"""Returns the density at point i that is attributable to a particle that is r_ij away."""
	return 	c_squared \
			* rho_0 \
			* (2.0/3 * quadratic(r=r_ij, h=h) * quadratic(r=0, h=h) - 1.0/3)

def r_ijs_density_and_pressure(i, sim_states, sim_params, computed_params):
	"""Returns the pairwise interagent distances, density and pressure at particle index i."""
	r_ijs = []
	rho = 0
	P = 0
	for particle in sim_states:
		r_ij = np.linalg.norm([
									particle['x_pos'] - sim_states[i]['x_pos'],
									particle['y_pos'] - sim_states[i]['y_pos'],
									particle['z_pos'] - sim_states[i]['z_pos'],
								])
		rho 	+= computed_params['m'] * quadratic(r=r_ij, h=sim_params['h'])
		P 		+= computed_params['c_squared']**2 \
					* computed_params['rho_0'] * \
					(2.0/3 * quadratic(r=r_ij, h=sim_params['h']) / quadratic(r=0, h=sim_params['h']) - 1.0/3)
		r_ijs.append(r_ij)
	return r_ijs, rho, P

def pairwise_direction(i, j, sim_states):
	"""Returns the norm-1 direction vector from particle i to particle j."""
	return normalize([[
									sim_states[j]['x_pos'] - sim_states[i]['x_pos'],
									sim_states[j]['y_pos'] - sim_states[i]['y_pos'],
									sim_states[j]['z_pos'] - sim_states[i]['z_pos'],
								]])

def pairwise_velocities(i, sim_states):
	"""Returns all pairwise velocities for particle i."""
	v_ijs = []
	for j in range(len(sim_states)):
		v_ijs.append(np.array([
						sim_states[j]['x_vel'] - sim_states[i]['x_vel'],
						sim_states[j]['y_vel'] - sim_states[i]['y_vel'],
						sim_states[j]['z_vel'] - sim_states[i]['z_vel']
		]))
	return v_ijs

# Define the force functions
def F_fluid(i, sim_states, sim_params, computed_params):
	"""Returns the hydrodynamic forces (pressure and viscosity) on particle index i in sim_states."""
	r_ijs, rho_i, P_i = r_ijs_density_and_pressure(i, sim_states, sim_params, computed_params)
	throwaway, rho_j, P_j = r_ijs_density_and_pressure(i, sim_states, sim_params, computed_params)
	v_ijs = pairwise_velocities(i, sim_states)
	F_T = 0
	for j in range(len(sim_states)):
		if i == j: continue
		r_ij = r_ijs[j]
		v_ij = v_ijs[j]
		F_T += - computed_params['m'] \
				* ( P_i / rho_i**2 + P_j / rho_j**2 ) \
				* quadratic_grad(r=r_ij, h=sim_params['h']) \
				* pairwise_direction(i, j, sim_states) \
				+ \
				computed_params['m'] * 2.0 \
				* computed_params['mu'] / (rho_i * rho_j) \
				* v_ij / r_ij \
				* quadratic_grad(r=r_ij, h=sim_params['h'])
	return F_T

# TO DO: insert updated mathematics once spline is ready
def F_attractor(i, sim_states, sim_params, computed_params):
	return np.zeros((3,))

# Define initial calculation of important constants
def compute_derived_parameters(
										p,
										M,
										h,
										gamma,
										rho_0,
										a_max,
										v_max,
										Re,
										inter_agent_w=1,
										attractor_w=1,
										obstacle_w=1
									):
	"""Computes and returns the parameters (mass, mu, squared information speed,
	and pressure) that can be derived from the given parameters."""
	# This mass is selected according to Song et al '17 OE to create a mixed
	# repelling and attracting force with a natural resting interparticle separation
	# distance of 0.5858 * h
	m = 2.0/3 * rho_0 / quadratic(0, p['h'])
	# these mu and c_squared are calculated for r=h assuming M=1 and
	# using a_max = || Re * (F_mu + F_P) ||
	# for the two particle case
	mu = a_max / v_max / (1 + Re) * 32 * math.pi / 15 * m * h**5 * quadratic(r=h, h=h)**2
	c_squared = a_max \
					/ (1 + 1/Re) \
					* (quadratic(r=0, h=h) + quadratic(r=h, h=h))**2 \
					/ ( quadratic_grad(r=h, h=h) * (2*quadratic(r=h, h=h) - quadratic(r=0, h=h)) )
	return {
		'm': m,
		'mu': mu,
		'c_squared': c_squared
	}

def mac_python(node_name, sim_states, sim_params):
	'''Computes the Multi-Agent Control (MAC) algorithm for one agent.

	Expects sim_states to contain x,y,z positions and velocities for all agents.
	Expects sim_params to contain: h, Re, a_max, v_max, x_target_pos, y_target_pos, z_target_pos.'''

	# Fixed parameters
	M = 1
	gamma = 1
	rho_0 = 1
	inter_agent_w = 1
	attractor_w = 1
	obstacle_w = 1 # not used in benchmark

	# Load data from JSON
	p = json.loads(sim_params)['sim_params']
	s = json.loads(sim_states)['sim_states']

	# Find self in data
	me = next(x for x in range(len(s)) if s[x]['agent_name'] == node_name)
	
	# Calculate important constants
	computed_params = compute_derived_parameters(
									p=p,
									M=M,
									h=p['h'],
									gamma=gamma,
									rho_0=rho_0,
									a_max=p['a_max'],
									v_max=p['v_max'],
									Re=p['Re'],
									inter_agent_w=inter_agent_w,
									attractor_w=attractor_w,
									obstacle_w=obstacle_w
	)
	computed_params['rho_0'] = rho_0

	# Calculate all pairwise force contributions
	accel = np.zeros((1,3))
	accel += inter_agent_w * F_fluid(me, s, p, computed_params)
	accel += attractor_w * F_attractor(me, s, p, computed_params)

	# Add pairwise force contributions to sim state
	s[me]['x_acc'] = accel[0,0]
	s[me]['y_acc'] = accel[0,1]
	s[me]['z_acc'] = accel[0,2]

	# Pack sim state back to JSON format and return
	return json.dumps(s)

## Basic test case
if __name__ == '__main__':
	test_states = '''
		{
		"sim_states": [
			{
			"a_att": -0.059451295862951926,
			"agent_name": "follower_1",
			"b_att": 0.6313554259579981,
			"c_att": -0.33784737388190933,
			"d_att": 0.69549624117942443,
			"node_name": "mothership",
			"t_acc": 59270.949710648223,
			"t_pos": 59270.949710648223,
			"t_vel": 59270.949710648223,
			"target_latitude": 0,
			"target_longitude": 0,
			"x_acc": 7.9793239868909893,
			"x_alpha": 0,
			"x_omega": 0.00037536013529212335,
			"x_pos": -6224890.2755624885,
			"x_thrust": 0,
			"x_torque": 0,
			"x_vel": -1633.0830316286583,
			"y_acc": -3.4307682344733443,
			"y_alpha": 0,
			"y_omega": 0.00082906087585329403,
			"y_pos": 2676495.7579533551,
			"y_thrust": 0,
			"y_torque": 0,
			"y_vel": -4201.7353647754753,
			"z_acc": -0.2221219420000671,
			"z_alpha": 0,
			"z_omega": 0.00066445406074924166,
			"z_pos": 172850.17766930754,
			"z_thrust": 0,
			"z_torque": 0,
			"z_vel": 6165.5832200356735
			},
			{
			"a_att": -0.16935785127266395,
			"agent_name": "follower_2",
			"b_att": 0.67795094371989151,
			"c_att": -0.23361545733955011,
			"d_att": 0.67610964659117812,
			"node_name": "daughter_01",
			"t_acc": 59270.949710648223,
			"t_pos": 59270.949710648223,
			"t_vel": 59270.949710648223,
			"target_latitude": 0,
			"target_longitude": 0,
			"x_acc": 8.6524890902684852,
			"x_alpha": 0,
			"x_omega": 9.770719677650105e-05,
			"x_pos": -6750096.0060347412,
			"x_thrust": 0,
			"x_torque": 0,
			"x_vel": -212.58284520205268,
			"y_acc": -0.74942210701775536,
			"y_alpha": 0,
			"y_omega": 0.00090440590972613042,
			"y_pos": 584725.67400556244,
			"y_thrust": 0,
			"y_torque": 0,
			"y_vel": -4507.7213659530298,
			"z_acc": -0.24724947680554785,
			"z_alpha": 0,
			"z_omega": 0.00066500857307542883,
			"z_pos": 192413.57262490701,
			"z_thrust": 0,
			"z_torque": 0,
			"z_vel": 6162.0689150611315
			},
			{
			"a_att": -0.27464352813515275,
			"agent_name": "follower_3",
			"b_att": 0.70736876798730342,
			"c_att": -0.12314225847976866,
			"d_att": 0.63955949113909805,
			"node_name": "daughter_02",
			"t_acc": 59270.949710648223,
			"t_pos": 59270.949710648223,
			"t_vel": 59270.949710648223,
			"target_latitude": 0,
			"target_longitude": 0,
			"x_acc": 8.448874352927584,
			"x_alpha": 0,
			"x_omega": -0.00018959018512271361,
			"x_pos": -6591361.1240350502,
			"x_thrust": 0,
			"x_torque": 0,
			"x_vel": 1231.7964376999396,
			"y_acc": 2.0074401125236196,
			"y_alpha": 0,
			"y_omega": 0.00088926450585965296,
			"y_pos": -1565977.7156856135,
			"y_thrust": 0,
			"y_torque": 0,
			"y_vel": -4346.434773545333,
			"z_acc": -0.27069517338684312,
			"z_alpha": 0,
			"z_omega": 0.00066557330921796377,
			"z_pos": 210591.23963140105,
			"z_thrust": 0,
			"z_torque": 0,
			"z_vel": 6158.4549100783197
			},
			{
			"a_att": -0.37263949923456846,
			"agent_name": "follower_4",
			"b_att": 0.71881178468853302,
			"c_att": -0.0092313230086780926,
			"d_att": 0.58682553154971517,
			"node_name": "daughter_03",
			"t_acc": 59270.949710648223,
			"t_pos": 59270.949710648223,
			"t_vel": 59270.949710648223,
			"target_latitude": 0,
			"target_longitude": 0,
			"x_acc": 7.3892231107864497,
			"x_alpha": 0,
			"x_omega": -0.00045778975407918988,
			"x_pos": -5764869.1062101023,
			"x_thrust": 0,
			"x_torque": 0,
			"x_vel": 2550.3199241220441,
			"y_acc": 4.5602137492938581,
			"y_alpha": 0,
			"y_omega": 0.0007851499084135697,
			"y_pos": -3557694.8608533377,
			"y_thrust": 0,
			"y_torque": 0,
			"y_vel": -3734.4245988819953,
			"z_acc": -0.29235427403831471,
			"z_alpha": 0,
			"z_omega": 0.00066609006662330826,
			"z_pos": 227414.40325906681,
			"z_thrust": 0,
			"z_torque": 0,
			"z_vel": 6155.0481203629579
			},
			{
			"a_att": -0.46087028680469622,
			"agent_name": "leader",
			"b_att": 0.71193973853458925,
			"c_att": 0.105218413321855,
			"d_att": 0.51929709505627819,
			"node_name": "daughter_04",
			"t_acc": 59270.949710648223,
			"t_pos": 59270.949710648223,
			"t_vel": 59270.949710648223,
			"target_latitude": 0,
			"target_longitude": 0,
			"x_acc": 5.5818531802414375,
			"x_alpha": 0,
			"x_omega": -0.00068005906017447792,
			"x_pos": -4354467.6443575686,
			"x_thrust": 0,
			"x_torque": 0,
			"x_vel": 3606.2647141271455,
			"y_acc": 6.6506949577445802,
			"y_alpha": 0,
			"y_omega": 0.00060247589319035132,
			"y_pos": -5188651.6569497762,
			"y_thrust": 0,
			"y_torque": 0,
			"y_vel": -2734.9527499245196,
			"z_acc": -0.3124396993806704,
			"z_alpha": 0,
			"z_omega": 0.00066650444956729696,
			"z_pos": 243051.36340820932,
			"z_thrust": 0,
			"z_torque": 0,
			"z_vel": 6152.1269267466769
			}
		]
		}
	'''
	test_params = json.dumps({
		'sim_params': {
			'h':1000,
			'Re':20,
			'a_max':1,
			'v_max':15,
			'x_target_pos':0,
			'y_target_pos':0,
			'z_target_pos':0
		}
	})
	r = mac_python(
						'follower_1',
						test_states,
						test_params
					)
	print("MAC function ran successfully. Test passed. Output:")
	print(r)