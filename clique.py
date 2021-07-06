import os
import networkx as nx
import JSON_IO as json
import aim_trajectory_picking.functions as func
import matplotlib.pyplot as plt

directory = r'.\basesets'
filename = 'base_test_0.txt'
fullpath = os.path.join(directory,filename)
list_of_trajectories = json.read_trajectory_from_json(fullpath)
G = func.transform_graph(list_of_trajectories)

visualize = True

def clique_set(G,weights=None,visualize=False):
    if visualize:
        plt.figure()
        nx.draw(G,with_labels=True)
        plt.show()
        
    clique, weights = nx.max_weight_clique(G,weights)
    C = func.transform_graph(clique)

    if visualize:
        plt.figure()
        nx.draw(C,with_labels=True)
        plt.show()

C, weights = clique_set(G,None,True)