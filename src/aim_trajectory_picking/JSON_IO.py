import json
import os
from aim_trajectory_picking import functions as func
import pickle


json_types = (list, dict, str, int, float, bool, type(None))

class PythonObjectEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, json_types):
            return super().default(self, obj)
        return {'_python_object': pickle.dumps(obj).decode('latin-1')}

def as_python_object(dct):
    if '_python_object' in dct:
        return pickle.loads(dct['_python_object'].encode('latin-1'))
    return dct

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
    with open(filename,'r') as file:
        input_data = json.loads(file.read(),object_hook=as_python_object)
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
        tra = func.Trajectory(trajectory["id"], trajectory["donor"], trajectory["target"], trajectory["value"])
        for collision in trajectory["collisions"]:
            tra.add_collision_by_id(collision)
        liste.append(tra)
    return liste


def read_trajectory_from_json_v2(filename):
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

    collisions: set
        set of collisions between trajectories
    '''
    input_data = read_data_from_json_file(filename)
    liste = []
    collision_ids = set()
    for trajectory in input_data["trajectories"]:
        tra = func.Trajectory(trajectory["id"], trajectory["donor"], trajectory["target"], trajectory["value"])
        liste.append(tra)
        for collision in trajectory["collisions"]:
            collision_ids.add((tra.id,collision))
            tra.add_collision_by_id(collision)
    
    collisions = [(liste[pair[0]], liste[pair[1]]) for pair in collision_ids]
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
        json.dump(data, outfile, cls=PythonObjectEncoder, sort_keys=False, indent=4)

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
        donor, target, trajectories = func.create_data(5,5,10,0.05)
        write_trajectory_to_json('datasets/dataset_'+str(i)+'.txt',trajectories)

def generate_increasing_datasets(num_datasets,increase):
    upper_limit_trajectories = 11000
    if increase**num_datasets > upper_limit_trajectories: # this is an assumption on the high end of expected number of trajectories
        print("Final datasets will become to large, fewer sets will be generated than the desired amount")

    donor = 0
    target = 0
    for i in range(1,num_datasets+1):
        if increase*i > upper_limit_trajectories:
            break
        donor = donor + 5*i
        target = target + 5*i
        print(increase*i)
        _, _, trajectories = func.create_data(donor,target,increase*i,0.05)
        write_trajectory_to_json('timesets/increasing_set_'+str(i)+'.json',trajectories)

def write_value_trajectories_runtime_from_file( combined_results,filename='results.txt',):
    for key1 in combined_results:
        for key2 in combined_results[key1]:
            combined_results[key1][key2]['trajectories'] = [e.__dict__ for e in combined_results[key1][key2]['trajectories']]
    write_data_to_json_file(filename,combined_results)

def read_value_trajectories_runtime_from_file(filename='results.txt'):
    input_data = read_data_from_json_file(filename)
    for key1 in input_data:
        for key2 in input_data[key1]:
            liste = []
            for trajectory in input_data[key1][key2]["trajectories"]:
                tra = func.Trajectory(trajectory["id"], trajectory["donor"], trajectory["target"], trajectory["value"])
                for collision in trajectory["collisions"]:
                    tra.add_collision_by_id(collision)
                liste.append(tra)
            input_data[key1][key2]["trajectories"] = liste
    return input_data

def generate_big_datasets():
    HIGH_DONORS = 50
    HIGH_TARGETS = 50
    LOW_DONORS = 5
    LOW_TARGETS = 5
    COLLISION_RATE = 0.05
    LOW_TRAJECTORIES = 500
    HIGH_TRAJECTORIES= 50000
    print('started generating data')
    _,_, trajectories = func.create_data(HIGH_DONORS, HIGH_TARGETS, LOW_TRAJECTORIES,COLLISION_RATE)
    write_trajectory_to_json('big_datasets/highD_highT_lowT.txt', trajectories)
    _,_, trajectories = func.create_data(HIGH_DONORS, LOW_TARGETS, LOW_TRAJECTORIES,COLLISION_RATE)
    write_trajectory_to_json('big_datasets/highD_lowT_lowT.txt', trajectories)
    _,_, trajectories = func.create_data(LOW_DONORS, HIGH_TARGETS, LOW_TRAJECTORIES,COLLISION_RATE)
    write_trajectory_to_json('big_datasets/lowD_highT_lowT.txt', trajectories)
    _,_, trajectories = func.create_data(LOW_DONORS, LOW_TARGETS, LOW_TRAJECTORIES,COLLISION_RATE)
    write_trajectory_to_json('big_datasets/lowD_lowT_lowT.txt', trajectories)

    print('started high collision rate')
    _,_, trajectories = func.create_data(HIGH_DONORS, HIGH_TARGETS, HIGH_TRAJECTORIES,COLLISION_RATE)
    write_trajectory_to_json('big_datasets/highD_highT_highT.txt', trajectories)
    _,_, trajectories = func.create_data(HIGH_DONORS, LOW_TARGETS, HIGH_TRAJECTORIES,COLLISION_RATE)
    write_trajectory_to_json('big_datasets/highD_lowT_highT.txt', trajectories)
    _,_, trajectories = func.create_data(LOW_DONORS, HIGH_TARGETS, HIGH_TRAJECTORIES,COLLISION_RATE)
    write_trajectory_to_json('big_datasets/lowD_highT_highT.txt', trajectories)
    _,_, trajectories = func.create_data(LOW_DONORS, LOW_TARGETS, HIGH_TRAJECTORIES,COLLISION_RATE)
    write_trajectory_to_json('big_datasets/lowD_lowT_highT.txt', trajectories)

if __name__ == '__main__':
    generate_big_datasets()
