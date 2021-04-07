
import json, math
import numpy as np

# Called from nodes being simulated
# Accepts as arguments:
#	a string for the name of the node
#	a json string of vector<sim_state>, which the python json module can automatically parse as a list of dicts
#	a json string of vector<sim_param>, which the python json module can automatically parse as a list of dicts

#	1) Read in the state vectors and parameters from cosmosstruc
#	2) Parse the data structure and modify the values in the python script
#	3) Return back the modified state vectors to the agent

def mac_python(node_name, sim_states, sim_params):
	'''Computes the Multi-Agent Control (MAC) algorithm for one agent.

	Expects sim_states to contain x,y,z positions and velocities. Expects sim_params to contain: h, Re, a_max, v_max.'''

	# Fixed parameters
	M = 1
	gamma = 1
	rho_0 = 1
	inter_agent_w = 1
	attractor_w = 1
	obstacle_w = 1 # not used in benchmark

	# Define the force functions
	# TO DO: insert updated mathematics once implemented from derivation (derivation in progress March 24)
	def pressure(i, j, sim_params): return np.zeros((3,))
	def viscosity(i, j, sim_params): return np.zeros((3,))
	def attractor(i, sim_params): return np.zeros((3,))

	# Define the function to combine all forces on an individual particle from other particles
	def inter_agent_interaction_force(i, j, sim_params):
		return pressure(i, j, sim_params) + viscosity(i, j, sim_params)

	# Load data from JSON
	p = json.loads(sim_params)['sim_params']
	s = json.loads(sim_states)['sim_states']

	# Find self in data
	me = next(x for x in range(len(s)) if s[x]['agent_name'] == node_name)

	# Define initial calculation of important constants
	def compute_derived_parameters(	p=p,
									M=M,
									gamma=gamma,
									rho_0=rho_0,
									inter_agent_w=inter_agent_w,
									attractor_w=attractor_w,
									obstacle_w=obstacle_w):
		"""Computes and returns the parameters (mass, mu, squared information speed, and pressure) that can be derived from the given parameters."""
		m = 2.0/3 * rho_0 / quadratic(0, p['h'])
		## TO DO: insert equations
		mu = 1
		c_squared = 1
		##
		P = c_squared \
			* rho_0 \
			* (2.0/3 * quadratic(r=p['h'], h=p['h']) * quadratic(r=0, h=p['h']) - 1.0/3)
		return m, mu, c_squared, P

	# Define kernel scaling factors
	## TO DO: insert more exact values from derivations
	omega_quadratic_3d = 1/math.pi
	omega_spline_3d = 1/math.pi
	##
	# Define the kernel functions
	# TO DO: insert B-spline kernel once derivation complete for 3D
	def quadratic(r, h):
		"""Implements a quadratic kernel function with support radius of 2h."""
		if r/h < 2:
			return 15.0 / (16 * math.pi * h**3) * ((r/h)**2 / 4 - r/h + 1)
		else:
			return 0

	def quadratic_grad(r, h):
		"""Implements the gradient of a quadratic kernel function with support radius of 2h."""
		if r/h < 2:
			return 15.0 / (16 * math.pi * h**4) * (r/h/2 - 1)
		else:
			return 0
	
	# Calculate important constants
	m, mu, c_squared, P = compute_derived_parameters(
									p=p,
									M=M,
									gamma=gamma,
									rho_0=rho_0,
									inter_agent_w=inter_agent_w,
									attractor_w=attractor_w,
									obstacle_w=obstacle_w
	)

	# Calculate all pairwise force contributions
	accel = np.zeros((3,))
	for i in range(len(s)):
		if me == i: continue
		accel += inter_agent_w * inter_agent_interaction_force(me, i, sim_params)
	accel += attractor_w * attractor(me, sim_params)

	# Add pairwise force contributions to sim state
	s[me]['x_acc'] = accel[0]
	s[me]['y_acc'] = accel[1]
	s[me]['z_acc'] = accel[2]

	# Pack sim state back to JSON format and return
	return json.dumps(s)

if __name__ == '__main__':
	test_states = '''
		{
		"sim_states": [
			{
			"a_att": -0.059451295862951926,
			"agent_name": "",
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
			"agent_name": "",
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
			"agent_name": "",
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
			"agent_name": "",
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
			"agent_name": "",
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
	r = mac_python('', test_states, '''{"sim_params":{"h":1}}''')
	print("MAC function ran successfully. Test passed. Output:")
	print(r)