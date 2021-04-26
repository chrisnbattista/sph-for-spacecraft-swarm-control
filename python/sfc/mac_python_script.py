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
	return 15.0 / (16 * math.pi * h**3)

def spline_scaling_factor_3d(h):
	"""Returns the scaling factor for the 3D version of the B-spline kernel. Ref: Mohaghan '92 and own derivations."""
	return 1.0/(math.pi * h**3)

# Define the kernel functions
def quadratic(r, h):
	"""Implements a quadratic kernel function with support radius of 2h."""
	if r/h < 2:
		return 	quadratic_scaling_factor_3d(h) \
				* ((r/h)**2 / 4 - r/h + 1)
	else:
		return 0

def d_quadratic_dr(r, h):
	"""Implements the gradient of a quadratic kernel function with support radius of 2h."""
	if r/h < 2:
		return 	quadratic_scaling_factor_3d(h) \
				* (r/h/2 - 1)
	else:
		return 0

def cubic_spline(r, h):
    '''Computes the cubic spline kernel with support radius of 2*h.'''
    if r > 2*h:
        return 0
    elif (r <= 2*h) and (r >= h):
        return spline_scaling_factor_3d(h) * 0.25 * (2 - r/h)**3
    else:
        return spline_scaling_factor_3d(h) * ( 1 - 1.5 * (r/h)**2 * (1 - r/h/2) ) 

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
				* d_quadratic_dr(r=r_ij, h=sim_params['h']) \
				* pairwise_direction(i, j, sim_states) \
				+ \
				computed_params['m'] * 2.0 \
				* computed_params['mu'] / (rho_i * rho_j) \
				* v_ij / r_ij \
				* d_quadratic_dr(r=r_ij, h=sim_params['h'])
	return F_T

def F_attractor(i, sim_states, sim_params, computed_params):
	"""Returns the force from the attractor (target) on agent i."""
	displacement = np.array([[
		sim_params['x_target_pos'] - sim_states[i]['x_pos'],
		sim_params['y_target_pos'] - sim_states[i]['y_pos'],
		sim_params['z_target_pos'] - sim_states[i]['z_pos'],
	]])
	r_itarget = np.linalg.norm(displacement)
	direction_itarget = normalize(displacement)
	return direction_itarget * cubic_spline(r_itarget, sim_params['h'])

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
					/ ( d_quadratic_dr(r=h, h=h) * (2*quadratic(r=h, h=h) - quadratic(r=0, h=h)) )
	return {
		'm': m,
		'mu': mu,
		'c_squared': c_squared
	}

def mac_python(node_name, sim_states, sim_params):
	'''Computes the Multi-Agent Control (MAC) algorithm for one agent.

	Expects sim_states to contain x,y,z positions and velocities for all agents.
	Expects sim_params to contain: h, Re, a_max, v_max, x_target_pos, y_target_pos, z_target_pos, accel_cap.'''

	# Fixed parameters
	M = 1
	gamma = 1
	rho_0 = 1
	inter_agent_w = 1
	attractor_w = 1
	obstacle_w = 1 # not used in benchmark

	# Load data from JSON
	p = json.loads(sim_params)['sim_params'][0] # @ Jim: added to follow your changes in master branch. is sim_params an array with one dictionary in it?
	s = json.loads(sim_states)['sim_states']

	# Find self in data
	me = next(x for x in range(len(s)) if s[x]['node_name'] == node_name)
	
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

	# Cap acceleration at physically reasonable value based on spacecraft capabilities.
	if(np.linalg.norm(accel)) > p['accel_cap']:
		accel = normalize(accel) * p['accel_cap']

	# Add pairwise force contributions to sim state
	s[me]['x_acc'] = accel[0,0]
	s[me]['y_acc'] = accel[0,1]
	s[me]['z_acc'] = accel[0,2]

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

	# @ Jim: I changed this to this interesting data structure as implied by the master-branch code you sent. Lmk if this was not correct.
	test_params = json.dumps(
		{
			'sim_params': [
				{
					'h':1000,
					'Re':20,
					'a_max':0.03,
					'v_max':15,
					'x_target_pos':0,
					'y_target_pos':0,
					'z_target_pos':0,
					'accel_cap':0.03
				}
			]
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