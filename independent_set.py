from aim_trajectory_picking.functions import greedy, random_choice, transform_graph
from JSON_IO import read_trajectory_from_json
import os
import matplotlib.pyplot as plt
import networkx as nx

visualize = True

directory = r'.\basesets'
filename = 'base_test_0.txt'
fullpath = os.path.join(directory,filename)
list_of_trajectories = read_trajectory_from_json(fullpath)
G = transform_graph(list_of_trajectories)

if visualize:
    plt.figure()
    nx.draw(G)
    plt.show()

# Empty initial set
I = nx.Graph()

# while G is not empty
while G.number_of_nodes() > 0:
    # Choose node from G
    nodes = list(G.nodes)
    choice = random_choice(nodes)
    print(choice)

    # Add node to set I
    I.add_node(choice)
    if False:
        plt.figure()
        nx.draw(G)
        plt.show()
    # Remove node from G
    '''The node seen to not get removed correctly, todo'''
    while G.number_of_edges(choice) > 0:
        G.remove_node(G.adjacency(choice))

    G.remove_node(choice)

# Return set I
if visualize:
    plt.figure()
    nx.draw(I)
    plt.show()
