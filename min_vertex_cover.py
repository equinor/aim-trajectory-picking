from clique import make_transformed_graph_from_trajectory_dictionaries
from networkx.algorithms.approximation import min_weighted_vertex_cover
import os

from networkx.classes.function import get_node_attributes
import aim_trajectory_picking.functions as func
import JSON_IO as jio
import networkx as nx
import matplotlib.pyplot as plt

if __name__ == '__main__':
    visualize = False
    
    # Read dataset
    directory = r'.\basesets'
    filename = 'base_test_0.txt'
    fullpath = os.path.join(directory,filename)
    nodes_list = jio.read_trajectory_from_json(fullpath)
    G = make_transformed_graph_from_trajectory_dictionaries(nodes_list)

    if visualize:
        plt.figure()
        nx.draw(G,with_labels=True)
        plt.show()

    vertex_cover = min_weighted_vertex_cover(G,weight='value')

    print(vertex_cover)
    print(nodes_list)