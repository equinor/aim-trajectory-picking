from networkx.algorithms.approximation.independent_set import maximum_independent_set
import os
from aim_trajectory_picking.functions import Trajectory, transform_graph
from JSON_IO import read_trajectory_from_json


# def translate_trajectory_objects_to_dictionaries(trajectories):
#     return [e.__dict__ for e in trajectories]

def translate_trajectory_objects_to_dictionaries(trajectories):
    node_set = []
    for element in trajectories:
        node_set.append(Trajectory(element.id, element.donor, element.target,element.value))
    dictionary = {}
    dictionary['value'] = sum(n.value for n in node_set)
    dictionary['trajectories'] = node_set
    return dictionary

if __name__ == '__main__':
    visualize = True

    # Read dataset
    directory = r'.\basesets'
    filename = 'base_test_0.txt'
    fullpath = os.path.join(directory,filename)
    nodes_list = read_trajectory_from_json(fullpath)
    G = transform_graph(nodes_list)

    # I think this does not solve the weighted problem
    max_ind_set = maximum_independent_set(G)
    for item in max_ind_set:
        print(item)
    print(max_ind_set)
    
    dicti = translate_trajectory_objects_to_dictionaries(max_ind_set)
    print(dicti)
    for item in dicti:
        print(item)