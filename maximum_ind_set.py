from networkx.algorithms.approximation.independent_set import maximum_independent_set
import os
from aim_trajectory_picking.functions import transform_graph
from JSON_IO import read_trajectory_from_json


visualize = True

# Read dataset
directory = r'.\basesets'
filename = 'base_test_0.txt'
fullpath = os.path.join(directory,filename)
nodes_list = read_trajectory_from_json(fullpath)
G = transform_graph(nodes_list)

max_ind_set = maximum_independent_set(G)
for item in max_ind_set:
    print(item)