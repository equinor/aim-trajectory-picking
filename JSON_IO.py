import json

from networkx.algorithms.asteroidal import create_component_structure
import aim_trajectory_picking.functions as dem
from numba import jit
# General function for reading a json-formatted .txt file
def read_data_from_json_file(filename):
    '''
    Reads data from a json-formatted text file.

    Parameters:
    -----------
    filename: str
        name of file to be read

    Returns:
    --------
    dictionary: dict
        a dictionary containing the JSON data read
    '''

    #add context manager? "with open(file) as y"
    file = open(filename,'r')
    input_data = json.loads(file.read())
    return input_data

# Wrapper for reading the trajectory from a json-formatted .txt file
def read_trajectory_from_json(filename):
    '''
    Wrapper for reading the trajectory from a json-formatted .txt file

    Paramenters:
    -----------
    filename: str
        name of file to be read
        
    Returns:
    --------
    liste: List<Trajectory>
        list of trajectory objects contained in filename
    
    '''
    input_data = read_data_from_json_file(filename)
    liste = []
    for trajectory in input_data["trajectories"]:
        tra = dem.Trajectory(trajectory["id"], trajectory["donor"], trajectory["target"], trajectory["value"])
        for collision in trajectory["collisions"]:
            tra.add_collision_by_id(collision)
        liste.append(tra)
    return liste

@jit(nopython=True)
def read_trajectory_from_json_v2(filename):
    input_data = read_data_from_json_file(filename)
    liste = []
    collisions = set()
    for trajectory in input_data["trajectories"]:
        tra = dem.Trajectory(trajectory["id"], trajectory["donor"], trajectory["target"], trajectory["value"])
        liste.append(tra)
        for collision in trajectory["collisions"]:
            collisions.add((tra.id,collision))
    return liste, collisions


# General function for writing data to a json-format .txt file
def write_data_to_json_file(filename, data):
    '''
    General function for writing data to a json-format .txt file

    Parameters:
    -----------
    filename: str
        name of file to be written to
    data: dictionary (usually)
        dictionary with information to be written to JSON format
    '''
    with open(filename, 'w') as outfile: 
        json.dump(data, outfile, sort_keys=False, indent=4)

# Wrapper function to write the trajectory in json-format to .txt file
def write_trajectory_to_json(filename,list_of_trajectories):
    '''
    Wrapper function to write the trajectory in json-format to .txt file

    Parameters:
    -----------
    filename: str
        name of file to be written to
    list_of_trajectories: List<Trajectory>
        list of trajectories to be written to fike

    Returns:
    --------
    None
    '''
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
    '''
    Function used once to generate datasets to be solved both by computer and by hand. DONT USE AGAIN
    '''
    for i in range(num_datasets):
        donor, target, trajectories = dem.create_data(5,5,10,0.05)
        write_trajectory_to_json('datasets/dataset_'+str(i)+'.txt',trajectories)

# # does not work atm due to json formatting needing a key
# results = {}
# lis = [32,20,26,31,23] # results of 5 greedy algorithms
# results['greedy'] = lis
# write_data_to_json_file('results.txt',results)


if __name__ == '__main__':
    HIGH_DONORS = 50
    HIGH_TARGETS = 50
    LOW_DONORS = 5
    LOW_TARGETS = 5
    COLLISION_RATE = 0.05
    LOW_TRAJECTORIES = 500
    HIGH_TRAJECTORIES= 50000
    print('started generating data')
    _,_, trajectories = dem.create_data(HIGH_DONORS, HIGH_TARGETS, LOW_TRAJECTORIES,COLLISION_RATE)
    write_trajectory_to_json('big_datasets/highD_highT_lowT.txt', trajectories)
    _,_, trajectories = dem.create_data(HIGH_DONORS, LOW_TARGETS, LOW_TRAJECTORIES,COLLISION_RATE)
    write_trajectory_to_json('big_datasets/highD_lowT_lowT.txt', trajectories)
    _,_, trajectories = dem.create_data(LOW_DONORS, HIGH_TARGETS, LOW_TRAJECTORIES,COLLISION_RATE)
    write_trajectory_to_json('big_datasets/lowD_highT_lowT.txt', trajectories)
    _,_, trajectories = dem.create_data(LOW_DONORS, LOW_TARGETS, LOW_TRAJECTORIES,COLLISION_RATE)
    write_trajectory_to_json('big_datasets/lowD_lowT_lowT.txt', trajectories)

    print('started high collision rate')
    _,_, trajectories = dem.create_data(HIGH_DONORS, HIGH_TARGETS, HIGH_TRAJECTORIES,COLLISION_RATE)
    write_trajectory_to_json('big_datasets/highD_highT_highT.txt', trajectories)
    _,_, trajectories = dem.create_data(HIGH_DONORS, LOW_TARGETS, HIGH_TRAJECTORIES,COLLISION_RATE)
    write_trajectory_to_json('big_datasets/highD_lowT_highT.txt', trajectories)
    _,_, trajectories = dem.create_data(LOW_DONORS, HIGH_TARGETS, HIGH_TRAJECTORIES,COLLISION_RATE)
    write_trajectory_to_json('big_datasets/lowD_highT_highT.txt', trajectories)
    _,_, trajectories = dem.create_data(LOW_DONORS, LOW_TARGETS, HIGH_TRAJECTORIES,COLLISION_RATE)
    write_trajectory_to_json('big_datasets/lowD_lowT_highT.txt', trajectories)