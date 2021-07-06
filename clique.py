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
print(list_of_trajectories[1])

visualize = True

def clique_set(G,weights=None,visualize=False):
    if visualize:
        plt.figure()
        posG = nx.get_node_attributes(G,'weight')
        print(posG)
        nx.draw(G,with_labels=True)
        plt.show()
        
    clique, weight = nx.max_weight_clique(G,weights)
    C = func.transform_graph(clique)

    if visualize:
        plt.figure()
        nx.draw(C,with_labels=True)
        plt.show()

weights = []
for i in range(G.number_of_nodes()):
    weights.append(list_of_trajectories[i].value)
print(weights)

C = clique_set(G,None,True)

