from numpy import true_divide
from aim_trajectory_picking import functions
import aim_trajectory_picking.functions as func
import networkx as nx
import matplotlib.pyplot as plt
import JSON_IO 
import os
import datasets
import numpy as np
import random

# donors1, targets1, trajectories1 = func.create_data(4, 4, 7, 0.04)
# print([n.value for n in trajectories1])
# test1 = func.transform_graph(trajectories1)
# plt.figure()
# nx.draw(test1, with_labels=True)
# plt.show()

# tra = func.abstract_trajectory_algorithm(test1, func.greedy,visualize=True)
# print(sum(n.value for n in tra))

# filename = "dataset1.txt"

def plot_performances(algorithms, results):
    plt.figure(figsize=(9,3))
    plt.subplot(121)
    #fig.title('Performance of various algorithms on trajectory problem')
    means = []
    algo_names = [e.__name__ for e in algorithms]
    dataset_names = [str(i) for i in range(len(results[algorithms[0].__name__]))]
    for algorithm in algorithms:
        results_per_dataset = [item['value'] for item in results[algorithm.__name__]]
        print(results_per_dataset)
        plt.plot(dataset_names, results_per_dataset, label=algorithm.__name__)
        means.append(np.mean(results_per_dataset))
    plt.legend()
    plt.subplot(122)
    plt.bar(algo_names, means)
    plt.show()

def create_results(algorithms, no_of_datasets):
    combined_results = {}
    for algorithm in test_functions:
        combined_results[algorithm.__name__] = []
    for i in range(no_of_datasets):
        donors, targets, trajectories = func.create_data(random.randint(1,10), random.randint(1,10), random.randint(50,500), 0.05)
        for algorithm in algorithms:
            answer = algorithm(trajectories, False)
            combined_results[algorithm.__name__].append(answer)
    return combined_results


if __name__ == '__main__':
    results = []
    directory = r'.\datasets'
    test_functions = [func.greedy_algorithm, func.NN_algorithm,func.random_algorithm, func.weight_transformation_algorithm, func.bipartite_matching_removed_collisions]
    combined_results = {}
    for algorithm in test_functions:
        combined_results[algorithm.__name__] = []
    dataset_names = []
    for filename in os.listdir(directory):
        dataset_names.append(filename)
        fullpath = os.path.join(directory,filename)
        # JSON_IO.write_data_to_json_file(filename, trajectories1)
        dataset1_after = JSON_IO.read_data_from_json_file(fullpath)
        print("Dataset: " + filename + "\n Performance:")
        #print([n.value for n in dataset1_after])
        #test1_1 = func.transform_graph(dataset1_after)
        
        # plt.figure()
        # nx.draw(test1_1, with_labels=True)
        # plt.show()
        #tra = func.abstract_trajectory_algorithm(test1_1, func.greedy,visualize=False)
        for algorithm in test_functions:
            answer = algorithm(dataset1_after, False)
            combined_results[algorithm.__name__].append(answer)
        #print(sum(n.value for n in tra))

        #results.append(sum([n.value for n in tra]))
    #read_results = JSON_IO.read_results('results.txt')
    #greedy_expected = read_results['greedy']
    #assert greedy_expected == results
    #print('done')
    # print(nx.is_isomorphic(test1, test1_1))
    r = create_results(test_functions, 5)

    #loop through all optimal trajectories found, check if collision
    for name in test_functions:
        for result in r[name.__name__]:
            if func.check_for_collisions(result['trajectories']):
                print("error in " + name.__name__)

    plot_performances(test_functions, r)

