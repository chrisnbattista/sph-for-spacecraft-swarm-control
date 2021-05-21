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
	"""Returns the scaling factor for the 3D version of the quadratic kernel. Ref: Song '17 and associated code."""
	return 15.0 / (16.0 * math.pi * (h**3))

def d_quadratic_scaling_factor_3d(h):
	'''Returns the derivative of the scaling factor for quadratic kernel.'''
	return 15.0 / (-3.0 * 16.0 * math.pi * (h**4))

def spline_scaling_factor_3d(h):
	"""Returns the scaling factor for the 3D version of the B-spline kernel. Ref: Mohaghan '92 and own derivations."""
	return 1.0/(math.pi * (h**3))

def d_spline_scaling_factor_3d(h):
	"""Returns the derivative of the scaling factor for the 3D version of the B-spline kernel. Ref: Mohaghan '92 and own derivations."""
	return -3.0/(math.pi * (h**4))

# Define the kernel functions
def quadratic(r, h):
	"""Implements a quadratic kernel function with support radius of 2h."""
	if r/h < 2:
		return 	quadratic_scaling_factor_3d(h) \
				* ((((r/h)**2) / 4) - (r/h) + 1)
	else:
		return 0

def d_quadratic_dr(r, h):
	"""Implements the gradient of a quadratic kernel function with support radius of 2h."""
	if r/h < 2:
		return 	d_quadratic_scaling_factor_3d(h) \
				* ((r/h/2) - 1)
	else:
		return 0

def cubic_spline(r, h):
	'''Computes the cubic spline kernel with support radius of 2*h.'''
	if r > 2*h:
		return 0
	elif (r <= 2*h) and (r >= h):
		return spline_scaling_factor_3d(h) \
			* 0.25 * ((2.0 - (r/h))**3)
	else:
		return spline_scaling_factor_3d(h) \
			* ( 1.0 - (1.5 * ((r/h)**2) * (1.0 - (r/h/2))) ) 

def d_cublic_spline_dr(r, h):
	'''Computes the gradient of the cubic spline kernel function.'''
	if r < h:
		return ( d_spline_scaling_factor_3d(h) \
			* ( (r/h) - ((3.0/4.0)*(r/h)**2) ) )
	elif r <= 2*h:
		return ( d_spline_scaling_factor_3d(h) \
			* ( (1/4.0)*((2-(r/h))**2) ) )
	else:
		return 0

# Define the property functions
def pairwise_density(c_squared, rho_0, r_ij, h):
	"""Returns the density at point i that is attributable to a particle that is r_ij away."""
	return 	c_squared \
			* rho_0 \
			* ((2.0/3.0) * (quadratic(r=r_ij, h=h) * quadratic(r=0, h=h)) - (1.0/3.0))

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
		P 		+= (computed_params['c_squared']**2) \
					* sim_params['rho_0'] \
					* ((2.0/3.0) * (quadratic(r=r_ij, h=sim_params['h']) / quadratic(r=0, h=sim_params['h'])) - (1.0/3.0))
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
				* ( (P_i / (rho_i**2)) + (P_j / (rho_j**2)) ) \
				* d_quadratic_dr(r=r_ij, h=sim_params['h']) \
				* pairwise_direction(i, j, sim_states) \
				+ \
				(computed_params['m'] * 2.0) \
				* (computed_params['mu'] / (rho_i * rho_j)) \
				* (v_ij / r_ij) \
				* d_quadratic_dr(r=r_ij, h=sim_params['h'])
	return F_T

def F_attractor(i, sim_states, sim_params, computed_params):
	"""Returns the force from the attractor (target) on agent i. Based on modification of reduced-density particle force from Song et al OE '17."""
	displacement = np.array([[
		sim_params['x_target_pos'] - sim_states[i]['x_pos'],
		sim_params['y_target_pos'] - sim_states[i]['y_pos'],
		sim_params['z_target_pos'] - sim_states[i]['z_pos'],
	]])
	r_iattractor = np.linalg.norm(displacement)
	direction_itarget = normalize(displacement)
	F = ( direction_itarget \
		* ((computed_params['c_a']**2) / cubic_spline(r=0, h=sim_params['h_attractor'])) \
		* d_cublic_spline_dr(r=r_iattractor, h=sim_params['h_attractor']) ) \
		* computed_params['m'] # because per OE '17 F_i is per unit mass
	return F

# Define initial calculation of important constants
def compute_derived_parameters(p):
	"""Computes and returns the parameters (mass, mu, squared information speed,
	and pressure) that can be derived from the given parameters in param dictionary p."""
	# This mass is selected according to Song et al '17 OE to create a mixed
	# repelling and attracting force with a natural resting interparticle separation
	# distance of 0.5858 * h
	m = (2.0/3.0) \
		 * (p['rho_0'] / quadratic(0, p['h']))
	# these mu and c_squared are calculated for r=h assuming M=1 and
	# using a_max = || Re * (F_mu + F_P) ||
	# for the two particle case
	mu = ((64.0* math.pi) / 135.0) \
		* (p['a_max'] / p['v_max']) \
		* (1.0 / (1.0 + p['Re'])) \
		* (p['rho_0'] / m) \
		* (p['h']**3) \
		* ((1 + (quadratic(r=p['h'], h=p['h']) / quadratic(r=0, h=p['h'])) )**2)
	c_squared = p['a_max'] \
					/ (1.0 + (1.0/p['Re'])) \
					* ((quadratic(r=0, h=p['h']) + quadratic(r=p['h'], h=p['h']))**2) \
					/ (d_quadratic_dr(r=p['h'], h=p['h']) * ( (2.0*quadratic(r=p['h'], h=p['h'])) - quadratic(r=0, h=p['h']) ) )
	# This is a new derivation, based on OE '17 but using a_max directly instead of turning radius calculations
	# It sets the attractor acceleration to half of a_max
	c_a = np.sqrt( (p['h_attractor'] / 6.0) * p['a_max'] )
	return {
		'm': m,
		'mu': mu,
		'c_squared': c_squared,
		'c_a': c_a
	}

def mac_python(node_name, sim_states, sim_params):
	'''Computes the Multi-Agent Control (MAC) algorithm for one agent.

	Expects sim_states to contain x,y,z positions and velocities for all agents.
	Expects sim_params to contain: h, h_attractor, Re, a_max, v_max, x_target_pos, y_target_pos, z_target_pos.'''

	# Load data from JSON
	p = json.loads(sim_params)['sim_params'] # @ Jim: adjusted per guidance
	if not 'h_attractor' in p: p['h_attractor'] = p['h']
	s = json.loads(sim_states)['sim_states']

	# Find self in data
	me = next(x for x in range(len(s)) if s[x]['node_name'] == node_name)
	
	# Calculate important constants
	computed_params = compute_derived_parameters(p=p)

	# Calculate all pairwise force contributions
	accel = np.zeros((1,3))
	accel += p['inter_agent_w'] * F_fluid(me, s, p, computed_params)
	a_F = p['attractor_w'] * p['a_max'] * F_attractor(me, s, p, computed_params)
	accel += a_F

	# Cap acceleration at physically reasonable value based on spacecraft capabilities.
	if(np.linalg.norm(accel)) > p['a_max']:
		accel = normalize(accel) * p['a_max']

	# Add pairwise force contributions to sim state
	s[me]['x_acc_diff'] = accel[0,0]
	s[me]['y_acc_diff'] = accel[0,1]
	s[me]['z_acc_diff'] = accel[0,2]

	# Pack sim state back to JSON format and return
	return json.dumps({'sim_states':s})

## Basic test case
if __name__ == '__main__':

	## Test a two-particle case
	sep = 1000
	test_states = json.dumps(
		{
			"sim_states": [
				{
					"node_name": "mothership",
					"x_acc": 0,
					"x_pos": 0,
					"x_vel": 0,
					"y_acc": 0,
					"y_pos": 0,
					"y_vel": 0,
					"z_acc": 0,
					"z_pos": 0,
					"z_vel": 0
				},
				{
					"node_name": "follower",
					"x_acc": 0,
					"x_pos": sep / 1.4,
					"x_vel": 0,
					"y_acc": 0,
					"y_pos": sep / 1.4,
					"y_vel": 0,
					"z_acc": 0,
					"z_pos": 0,
					"z_vel": 0
				}
			]
		}
	)

	# @ Jim: changed to match guidance
	test_params = json.dumps(
		{
			'sim_params':
				{
					'internode_distance':5000, # @Jim, Eric, Zhuoyuan – should this replace h?
					'Re':20,
					'a_max':0.03,
					'v_max':15,
					'M':1,
					'gamma':1,
					'rho_0':1,
					'inter_agent_w':1,
					'attractor_w':1,
					'obstacle_w':1,
					'x_target_pos':0, # should be target position in string of pearls (different per agent)
					'y_target_pos':0, # should be target position in string of pearls (different per agent)
					'z_target_pos':0 # should be target position in string of pearls (different per agent)
				}
		}
	)
	follower_state = mac_python(
						'follower',
						test_states,
						test_params
					)
	mothership_state = mac_python(
						'mothership',
						test_states,
						test_params
					)

	print("MAC function ran successfully. Test passed. State (follower):")
	print(follower_state)
	print("State (mothership):")
	print(mothership_state)
