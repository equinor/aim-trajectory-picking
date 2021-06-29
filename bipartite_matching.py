import aim_trajectory_picking.functions as func
#import aim_trajectory_picking.functions.Trajectory
import networkx as nx

# Generate random test set
donors, targets, trajectories = func.create_data(15,15,1000,0.05)

G = nx.Graph()
print(type(trajectories))
G.add_nodes_from(trajectories)
# Add collisions from donors and targets
for i in range(len(trajectories)):
    for j in range(i, len(trajectories)):
        if i != j:
            if trajectories[i].id in  trajectories[j].collisions:
                G.add_edge(trajectories[i], trajectories[j])

# TODO 4 times transform graph? 

optimal_trajectories = func.abstract_trajectory_algorithm(G,func.greedy, False)

transformed_graph = func.transform_graph(trajectories)
optimal_trajectories_greedy = func.abstract_trajectory_algorithm(transformed_graph,func.greedy, False)
transformed_graph = func.transform_graph(trajectories)
func.abstract_trajectory_algorithm(transformed_graph, func.random_choice, False)
transformed_graph = func.transform_graph(trajectories)
func.abstract_trajectory_algorithm(transformed_graph, func.NN_transformation, False)

bi_graph = func.bipartite_graph(donors, targets,optimal_trajectories)
matching = nx.max_weight_matching(bi_graph)
#print(matching)
matching_sum = 0
for donor, target in matching:
    matching_sum += bi_graph[donor][target]['weight']
#print("Greedy answer sum: " + str(sum(n.value for n in optimal_trajectories_greedy)))
print("Bipartite matching sum: " + str(matching_sum))
