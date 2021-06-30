import aim_trajectory_picking.functions as func
import copy
import matplotlib.pyplot as plt
import numpy as np

data_1 = func.create_data(5,5,100,0.05)
data_2 = func.create_data(5,5,500,0.05)

g1 = func.transform_graph(data_1[2])
g2 = func.transform_graph(data_2[2])

greedy_answer_g1 = func.abstract_trajectory_algorithm(copy.deepcopy(g1), func.greedy, False)
greedy_answer_g2 = func.abstract_trajectory_algorithm(copy.deepcopy(g2), func.greedy, False)

weight_trans_answer_g1 = func.abstract_trajectory_algorithm(copy.deepcopy(g1), func.weight_transformation, False)
weight_trans_answer_g2 = func.abstract_trajectory_algorithm(copy.deepcopy(g2), func.weight_transformation, False)

greedy = [sum(n.value for n in greedy_answer_g1), sum(n.value for n in greedy_answer_g2)]
weight = [sum(n.value for n in weight_trans_answer_g1), sum(n.value for n in weight_trans_answer_g2)]
datasets = ['g1', 'g2']
algos = ['greedy', 'weight']

plt.figure(figsize=(9,3))
plt.subplot(121)
#fig.title('Performance of various algorithms on trajectory problem')
plt.plot(datasets, greedy, label='greedy', color='k')
plt.plot(datasets, weight,  label='weight trans', color='b')
plt.legend()
plt.subplot(122)
plt.bar(algos, [np.mean(greedy), np.mean(weight)])
plt.show()
