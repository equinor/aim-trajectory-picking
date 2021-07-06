from aim_trajectory_picking.functions import greedy, random_choice, transform_graph
from JSON_IO import read_trajectory_from_json
import os
import matplotlib.pyplot as plt
import networkx as nx

visualize = True

# Read dataset
directory = r'.\basesets'
filename = 'base_test_0.txt'
fullpath = os.path.join(directory,filename)
nodes_list = read_trajectory_from_json(fullpath)
G = transform_graph(nodes_list)


def single_independent_set(G,visualize=False):
    
    # Initial graph
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
        if visualize:
            plt.figure()
            nx.draw(G)
            plt.show()
        # Remove node from G
        '''The node seen to not get removed correctly, todo'''
        
        for n in list(G.neighbors(choice)):
            G.remove_node(n)
        G.remove_node(choice)

    # Return set I
    if visualize:
        plt.figure()
        nx.draw(I)
        plt.show()
    
    return I

I = single_independent_set(G,True)