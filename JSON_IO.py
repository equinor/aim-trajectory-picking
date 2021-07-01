import json

from networkx.algorithms.asteroidal import create_component_structure
import aim_trajectory_picking.functions as dem

def read_data_from_json_file(filename):
    file = open(filename,'r')
    input_data = json.loads(file.read())
    liste = []
    for trajectory in input_data["trajectories"]:
        #print(trajectory)
        tra = dem.Trajectory(trajectory["id"], trajectory["donor"], trajectory["target"], trajectory["value"])
        for collision in trajectory["collisions"]:
            tra.add_collision_by_id(collision)
        liste.append(tra)
    return liste

def write_data_to_json_file(filename, list_of_trajectories):
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
    with open(filename, 'w') as outfile: 
        json.dump(JSON_trajectories, outfile, sort_keys=False, indent=4)

# "temp" for generating datasets
# for i in range(5):
#     donor, target, trajectories = dem.create_data(5,5,10,0.05)
#     write_data_to_json_file('datasets/dataset_'+str(i)+'.txt',trajectories)

# does not work atm due to json formatting needing a key
lis = [32,20,26,31,23] # results of 5 greedy algorithms
with open('results.txt','w') as filename:
    json.dumps(lis,filename,sort_keys=False)