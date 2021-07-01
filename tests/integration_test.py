from aim_trajectory_picking import functions
import aim_trajectory_picking.functions as func
import networkx as nx
import matplotlib.pyplot as plt
import JSON_IO 

donors1, targets1, trajectories1 = func.create_data(4, 4, 7, 0.04)
print([n.value for n in trajectories1])
test1 = func.transform_graph(trajectories1)
plt.figure()
nx.draw(test1, with_labels=True)
plt.show()

tra = func.abstract_trajectory_algorithm(test1, func.greedy,visualize=True)
print(sum(n.value for n in tra))
'''
filename = "dataset1.txt"
JSON_IO.write_data_to_json_file(filename, trajectories1)
dataset1_after = JSON_IO.read_data_from_json_file(filename)

test1_1 = func.transform_graph(dataset1_after)
plt.figure()
nx.draw(test1_1, with_labels=True)
plt.show()

print(nx.is_isomorphic(test1, test1_1))
'''