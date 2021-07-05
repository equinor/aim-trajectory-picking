import json

from networkx.algorithms.asteroidal import create_component_structure
import aim_trajectory_picking.functions as dem

# General function for reading a json-formatted .txt file
def read_data_from_json_file(filename):
    file = open(filename,'r')
    input_data = json.loads(file.read())
    return input_data

# Wrapper for reading the trajectory from a json-formatted .txt file
def read_trajectory_from_json(filename):
    input_data = read_data_from_json_file(filename)
    liste = []
    for trajectory in input_data["trajectories"]:
        tra = dem.Trajectory(trajectory["id"], trajectory["donor"], trajectory["target"], trajectory["value"])
        for collision in trajectory["collisions"]:
            tra.add_collision_by_id(collision)
        liste.append(tra)
    return liste

# General function for writing data to a json-format .txt file
def write_data_to_json_file(filename, data):
    with open(filename, 'w') as outfile: 
        json.dump(data, outfile, sort_keys=False, indent=4)

# Wrapper function to write the trajectory in json-format to .txt file
def write_trajectory_to_json(filename,list_of_trajectories):
    JSON_trajectories = {}
    JSON_trajectories['trajectories'] = []
    for x in range(len(list_of_trajectories)):
        trajectory = {}
        trajectory['id'] = list_of_trajectories[x].id
        trajectory['donor'] = list_of_trajectories[x].donor
        trajectory['target'] = list_of_trajectories[x].target
        trajectory['value'] = list_of_trajectories[x].value
        trajectory['collisions'] = list_of_trajectories[x].collisions
        JSON_trajectories['trajectories'].append(trajectory)
    write_data_to_json_file(filename,JSON_trajectories)

# For generating datasets with choosen parameters
def generate_datasets_as_json_files(num_datasets):
    for i in range(num_datasets):
        donor, target, trajectories = dem.create_data(5,5,10,0.05)
        write_trajectory_to_json('datasets/dataset_'+str(i)+'.txt',trajectories)

# does not work atm due to json formatting needing a key
results = {}
lis = [32,20,26,31,23] # results of 5 greedy algorithms
results['greedy'] = lis
write_data_to_json_file('results.txt',results)