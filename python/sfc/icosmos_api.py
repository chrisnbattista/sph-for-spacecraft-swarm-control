import numpy as np
import json, copy
from multi_agent_kinetics import worlds, forces

## Utility functions

def icosmos_to_mac(json_transmission):
    """Convert a JSON iCOSMOS state object into a usable format for MAC."""
    
    converted_json = json.loads(json_transmission)
    mac_format_data = np.zeros( (len(converted_json['sim_states']), 9) )
    for agent_i in range(len(converted_json['sim_states'])):
        mac_format_data[agent_i,3] = converted_json['sim_states'][agent_i]['x_pos']
        mac_format_data[agent_i,4] = converted_json['sim_states'][agent_i]['y_pos']
        mac_format_data[agent_i,5] = converted_json['sim_states'][agent_i]['z_pos']
        mac_format_data[agent_i,6] = converted_json['sim_states'][agent_i]['x_vel']
        mac_format_data[agent_i,7] = converted_json['sim_states'][agent_i]['y_vel']
        mac_format_data[agent_i,8] = converted_json['sim_states'][agent_i]['z_vel']
        mac_format_data[:,1] = range(mac_format_data.shape[0])
        mac_format_data[:,2] = 10
    
    return mac_format_data

def mac_to_icosmos(mac_representation, original_icosmos_representation):
    """Overlays the data in a MAC matrix into an iCOSMOS JSON-style representation and returns the result."""
    modified_icosmos_representation = copy.deepcopy(original_icosmos_representation)
    for agent_i in range(mac_representation.shape[0]):
        modified_icosmos_representation['sim_states'][agent_i]['x_pos'] = mac_representation[agent_i,3]
        modified_icosmos_representation['sim_states'][agent_i]['y_pos'] = mac_representation[agent_i,4]
        modified_icosmos_representation['sim_states'][agent_i]['z_pos'] = mac_representation[agent_i,5]
        modified_icosmos_representation['sim_states'][agent_i]['x_vel'] = mac_representation[agent_i,6]
        modified_icosmos_representation['sim_states'][agent_i]['y_vel'] = mac_representation[agent_i,7]
        modified_icosmos_representation['sim_states'][agent_i]['z_vel'] = mac_representation[agent_i,8]
    return modified_icosmos_representation

def apply_mac(icosmos_json, pressure=10e10, spacing=10e6):
    """Applies the Smoothed Particle Hydrodynamics MAC system to a state transmission from iCOSMOS."""

    ## Convert data into MAC representation
    converted_test_data = icosmos_to_mac(icosmos_json)
    mac_world = worlds.World(
        spatial_dims=3,
        n_agents=converted_test_data.shape[0],
        control_agents=[None]*converted_test_data.shape[0],
        initial_state=converted_test_data,
        timestep=1,
        forces=[lambda w,c: forces.world_pressure_force(world=w, pressure=pressure, h=spacing/0.5858, context=c)]
    )

    ## Apply HCL (more infomation needed from iCOSMOS / sensors)
    pass

    ## Compute the MAC
    mac_world.advance_state()
    mac_state = mac_world.get_state()

    ## Add in the acceleration terms as a command signal
    ## Convert the new state to the original JSON format
    mac_translated_to_icosmos = mac_to_icosmos(mac_state, json.loads(icosmos_json))
    delta_v = (mac_state - converted_test_data)[:,6:9]
    for agent_i in range(mac_state.shape[0]):
        mac_translated_to_icosmos['sim_states'][agent_i]['x_acc'] = delta_v[agent_i,0]
        mac_translated_to_icosmos['sim_states'][agent_i]['y_acc'] = delta_v[agent_i,1]
        mac_translated_to_icosmos['sim_states'][agent_i]['z_acc'] = delta_v[agent_i,2]
    json_return_transmission = json.dumps(mac_translated_to_icosmos)
    return json_return_transmission