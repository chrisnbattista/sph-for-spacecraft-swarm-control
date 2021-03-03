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

## Test data

test_data = '''
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

## Convert data into MAC representation

converted_test_data = icosmos_to_mac(test_data)
mac_world = worlds.World(
    spatial_dims=3,
    n_agents=converted_test_data.shape[0],
    control_agents=[None]*converted_test_data.shape[0],
    initial_state=converted_test_data,
    timestep=1,
    forces=[lambda w,c: forces.world_pressure_force(world=w, pressure=10000, h=10e6, context=c)]
)

## Compute the MAC

mac_world.advance_state()

## Output the new state to console and demonstrate delta_v

test_data_with_mac_applied = mac_world.get_state()
print((test_data_with_mac_applied - converted_test_data)[:,6:9])

## Convert the new state to the original JSON format
json_return_transmission = json.dumps(mac_to_icosmos(mac_world.get_state(), json.loads(test_data)))
print(json_return_transmission)