# Wraps the COSMOS-compatible control routines for use in the Python testbed.

import json
import numpy as np
import mac_python_script
from multi_agent_kinetics import worlds

def mak_to_cosmos_state(mak_world, translation_table):
    '''Converts a multi-agent-kinetics world state to an iCOSMOS compatible JSON string. Only works for the standard MAK 3d schema.'''
    data = {'sim_states':[]}
    mak_state = mak_world.get_state()
    for row in mak_state:
        pos = row[worlds.pos[3]] # 3 for 3D
        data['sim_states'].append(
            {
                'node_name':translation_table['id'][row[worlds.ID[3]]], # look up the node_name in the foreign key table provided, based on the ID in the row
                'x_pos':pos[0],
                'y_pos':pos[1],
                'z_pos':pos[2]
            }
        )
    return json.dumps(data)

def cosmos_to_mak_state(cosmos_world):
    '''Converts iCOSMOS-compatible JSON string to a multi-agent-kinetics world state array. Only compatible with 3D.'''
    data = json.loads(cosmos_world)['sim_states']
    return data

class SPHController:
    def __init__(self, name, params, translation_table):
        '''params = dictionary to JSONify and send to mac_python function calls as sim_params.
            name = node_name for referencing iCOSMOs data.
            translation_table = a mapping between MAK columns and iCOSMOS fields.'''
        self.name = name
        self.params = params
        self.jsonified_params = json.dumps(params)
        self.translation_table = translation_table
    def control(self, world):
        '''Returns forces to exert on controlled agent based on world state.'''
        cosmos_state = mak_to_cosmos_state(world.state, self.translation_table)
        modified_cosmos_state = mac_python_script.mac_python(self.name, cosmos_state, self.jsonified_params)
        control_agent_cosmos_state = next((x for x in modified_cosmos_state['sim_states'] if x['node_name']==self.name))
        diff_accel = np.array((
            control_agent_cosmos_state['x_acc_diff'],
            control_agent_cosmos_state['y_acc_diff'],
            control_agent_cosmos_state['z_acc_diff'],
        ))
        return diff_accel