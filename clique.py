import os
import networkx as nx
import JSON_IO as json
import aim_trajectory_picking.functions as func
import matplotlib.pyplot as plt

def translate_trajectory_objects_to_dictionaries(trajectories):
    new_list = []
    for item in trajectories:
        new_list.append(item.__dict__)
    return new_list

# def translate_trajectory_objects_to_dictionaries(trajectories):
#     return [e.__dict__ for e in trajectories]

def make_transformed_graph_from_trajectory_dictionaries(trajectory_dictionaries):
    G = nx.Graph()
    ids = [e['id'] for e in trajectory_dictionaries]
    G.add_nodes_from(ids)
    node_attrs = {}
    for i in range(len(trajectory_dictionaries)):
        node_attrs[ids[i]] = trajectory_dictionaries[i]
    nx.set_node_attributes(G,node_attrs)
    for i in range(len(trajectory_dictionaries)):
        for j in range(i, len(trajectory_dictionaries)):
            if i != j:
                if mutually_exclusive_trajectories_dictionary(trajectory_dictionaries[i], trajectory_dictionaries[j]):
                    G.add_edge(trajectory_dictionaries[i]['id'], trajectory_dictionaries[j]['id'])
    return G

# def make_transformed_graph_from_trajectory_dictionaries(trajectory_dictionaries,values):
#     G = nx.Graph()
#     ids = [e['id'] for e in trajectory_dictionaries]
#     G.add_nodes_from(ids)
#     for i in range(len(trajectory_dictionaries)):
#         nx.set_node_attributes(G,values, trajectory_dictionaries[i])
#     for i in range(len(trajectory_dictionaries)):
#         for j in range(i, len(trajectory_dictionaries)):
#             if i != j:
#                 if mutually_exclusive_trajectories_dictionary(trajectory_dictionaries[i], trajectory_dictionaries[j]):
#                     G.add_edge(trajectory_dictionaries[i], trajectory_dictionaries[j])
#     return G

# def make_transformed_graph_from_trajectory_dictionaries(trajectory_dictionaries):
#     G = nx.Graph()
#     G.add_nodes_from(trajectory_dictionaries)
#     for i in range(len(trajectory_dictionaries)):
#         for j in range(i, len(trajectory_dictionaries)):
#             if i != j:
#                 if mutually_exclusive_trajectories_dictionary(trajectory_dictionaries[i], trajectory_dictionaries[j]):
#                     G.add_edge(trajectory_dictionaries[i], trajectory_dictionaries[j])
#     return G
 
def mutually_exclusive_trajectories_dictionary(t1, t2):
    if t1['donor'] == t2['donor']:
        return True
    elif t1['target'] == t2['target']:
        return True
    elif t1['id'] in t2['collisions']:
        return True
    return False

directory = r'.\basesets'
filename = 'base_test_0.txt'
fullpath = os.path.join(directory,filename)
list_of_trajectories = json.read_trajectory_from_json(fullpath)

values = []
for i in range(len(list_of_trajectories)):
    values.append(list_of_trajectories[i].value)

trajectories_dict = translate_trajectory_objects_to_dictionaries(list_of_trajectories)
print(trajectories_dict)
print(trajectories_dict[1])
G = make_transformed_graph_from_trajectory_dictionaries(trajectories_dict)



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

weights = 'value'
C, weights = clique_set(G,weights,True)


