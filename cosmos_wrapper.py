# Wraps the COSMOS-compatible control routines for use in the Python testbed.

import json
import numpy as np
from copy import deepcopy
import mac_python_script
from multi_agent_kinetics import worlds

def mak_to_cosmos_state(mak_state, translation_table):
    '''Converts a multi-agent-kinetics world state to an iCOSMOS compatible JSON string. Only works for the standard MAK 3d schema.'''
    data = {'sim_states':[]}
    for row in mak_state:
        pos = row[worlds.pos[3]] # 3 for 3D
        vel = row[worlds.vel[3]]
        data['sim_states'].append(
            {
                'node_name':translation_table['id'][row[worlds.ID[3]]], # look up the node_name in the foreign key table provided, based on the ID in the row
                'x_pos':pos[0],
                'y_pos':pos[1],
                'z_pos':pos[2],
                'x_vel':vel[0],
                'y_vel':vel[1],
                'z_vel':vel[2],
            }
        )
    return json.dumps(data)

def cosmos_to_mak_state(cosmos_state):
    '''Converts iCOSMOS-compatible JSON string to a multi-agent-kinetics world state array. Only compatible with 3D.'''
    data = json.loads(cosmos_state)['sim_states']
    return data

class SPHController:
    def __init__(self, name, params, translation_table, attractor=None):
        '''params = dictionary to JSONify and send to mac_python function calls as sim_params.
            name = node_name for referencing iCOSMOs data.
            translation_table = a mapping between MAK columns and iCOSMOS fields.'''
        self.name = name
        self.params = deepcopy(params)
        self.translation_table = translation_table
        self.attractor = attractor

    def control(self, world, context):
        '''Returns forces to exert on controlled agent based on world state.'''
        s = world.get_state()
        p = self.attractor(s[0,worlds.time[3]])
        self.params['sim_params']['x_attractor'] = p[0]
        self.params['sim_params']['y_attractor'] = p[1]
        self.params['sim_params']['z_attractor'] = p[2]
        cosmos_state = mak_to_cosmos_state(s, self.translation_table)
        modified_cosmos_state = json.loads(
            mac_python_script.mac_python(
                node_name=self.name,
                sim_states=cosmos_state,
                sim_params=json.dumps(self.params)
                )
        )
        control_agent_cosmos_state = next((x for x in modified_cosmos_state['sim_states'] if x['node_name']==self.name))
        diff_accel = np.array((
            control_agent_cosmos_state['x_acc_diff'],
            control_agent_cosmos_state['y_acc_diff'],
            control_agent_cosmos_state['z_acc_diff'],
        ))
        return diff_accel