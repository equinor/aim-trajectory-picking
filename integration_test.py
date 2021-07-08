from numpy import true_divide
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

def plot_performances(algorithms, results,_dataset_names=0):
    '''
    Fully automatic function that plots the results per algorithm. 

    Important: the names of the algorithms must be keys in the results dictionary, and every value is a list \
        that consists of dictionaries, which again contain the value of trajectories of that specific algorithm on that \
            specific dataset.
    
    Parameters:
    -----------
    algorithms: List<Function>
        list of functions used to obtain results

    results: dictionary{
        'algorithm1.__name__' : list<dictionary> [
            dictionary1{
                'value' : int  
                'trajectories': List<Trajectory>
            }, 
            dictionary2{
                'value' : int  
                'trajectories': List<Trajectory>
            },
            ...
        ], 
        'algorithm2.__name__' : list<dictionary> [
            dictionary1{
                'value' : int  
                'trajectories': List<Trajectory>
            }, 
            dictionary2{
                'value' : int  
                'trajectories': List<Trajectory>
            },
            ...
        ], 
        ...
    }

    Returns:
    --------
    None
    '''
    plt.figure(figsize=(9,3))
    plt.subplot(121)
    #fig.title('Performance of various algorithms on trajectory problem')
    means = []
    if _dataset_names == 0:
        dataset_names = [str(i) for i in range(len(results[algorithms[0].__name__]))]
    else:
        dataset_names = _dataset_names

    algo_names = [e.__name__ for e in algorithms]
    for algorithm in algorithms:
        results_per_dataset = [item['value'] for item in results[algorithm.__name__]]
        print(results_per_dataset)
        plt.plot(dataset_names, results_per_dataset, label=algorithm.__name__) 
        means.append(np.mean(results_per_dataset))
    plt.legend()
    plt.xticks(rotation=45)
    plt.subplot(122)
    plt.bar(algo_names, means)
    plt.xticks(rotation=45)
    plt.show()

'''
Sketch for making new plot, with one figure with one histogram per dataset, with one bar for every algorithm

dataset_names = []
for filename in os.listdir(directory):
    dataset_names.append(filename)


Trenger en dictionary med algoritmenavn som key, og resultat som values
Noe sånt: 

results_dict = {}
for algorithm in algorithms: 
    results_dict[algorithm.__name__ ] = 0

plotdata = pd.DataFrame({
    "algorithm1":[40, 12, 10, 26, 36],
    "algorithm2":[19, 8, 30, 21, 38],
    "algorithm3":[10, 10, 42, 17, 37]
    }, 
    index=dataset_names
)
plotdata.plot(kind="bar")
plt.title("Performance of Algorithms on Datasets")
plt.xlabel("Dataset")
plt.ylabel("Value")
'''

def create_results(algorithms, no_of_datasets):
    combined_results = {}
    print("progress:")
    print('#'*1)
    for algorithm in test_functions:
        combined_results[algorithm.__name__] = []
    print("hei")
    for i in range(no_of_datasets):
        print("er du serr")
        donors, targets, trajectories = func.create_data(random.randint(1,15), random.randint(1,15), random.randint(50,1000), 0.05)
        for algorithm in algorithms:
            print("started test " + str(i) + " with algorithm " + algorithm.__name__)
            answer = algorithm(trajectories, False)
            combined_results[algorithm.__name__].append(answer)
            print("done with test " + str(i) + " with algorithm " + algorithm.__name__)
    return combined_results

def create_data(no_of_datasets):
    '''
    Helper function to create datasets randomly for testing.

    Uses functions.create_data internally.

    Parameters:
    -----------
    no_of_datasets: int
        number of datasets desired
    
    Returns:
    --------
    data: List<List<Trajectories>>
        list of trajectory lists, aka multiple datasets.
    '''
    data = []
    for i in range(no_of_datasets):
        donors, targets, trajectories = func.create_data(random.randint(1,10), random.randint(1,10), random.randint(50,500), 0.05)
        data.append(trajectories)
    return trajectories

def calculate_results(algorithms, datasets):
    combined_results = {}
    for algorithm in algorithms:
        combined_results[algorithm.__name__] = []
    for data in datasets:
        for algorithm in algorithms:
            answer = algorithm(data, False)
            combined_results[algorithm.__name__].append(answer)
    return results
    


if __name__ == '__main__':
    results = []
    directory = r'.\datasets'
    test_functions = [func.greedy_algorithm, func.NN_algorithm,func.random_algorithm,
                     func.weight_transformation_algorithm, func.bipartite_matching_removed_collisions,
                     func.lonely_target_algorithm, func.reversed_greedy, func.invert_and_clique]
    combined_results = {}
    for algorithm in test_functions:
        combined_results[algorithm.__name__] = []
    dataset_names = []
    for filename in os.listdir(directory):
        dataset_names.append(filename)
        fullpath = os.path.join(directory,filename)
        # JSON_IO.write_data_to_json_file(filename, trajectories1)
        dataset1_after = JSON_IO.read_trajectory_from_json(fullpath)
        print("read file: " +filename)
        #print("Dataset: " + filename + "\n Performance:")
        #print([n.value for n in dataset1_after])
        #test1_1 = func.transform_graph(dataset1_after)
        
        # plt.figure()
        # nx.draw(test1_1, with_labels=True)
        # plt.show()
        #tra = func.abstract_trajectory_algorithm(test1_1, func.greedy,visualize=False)
        for algorithm in test_functions:
            answer = algorithm(dataset1_after, False)
            combined_results[algorithm.__name__].append(answer)
            print("done with algorithm: " + algorithm.__name__ + " on dataset: " + filename)


        #results.append(sum([n.value for n in tra]))
    for i in range(5):
        for algorithm in test_functions:
            print(algorithm.__name__ + " on " + dataset_names[i] + " gave result: " + str(combined_results[algorithm.__name__][i]['value']))
    for name in test_functions:
        for result in combined_results[name.__name__]:
            if func.check_for_collisions(result['trajectories']):
                print("error in " + name.__name__)

    plot_performances(test_functions,combined_results, dataset_names)
    # for i in range(5):
    #     print("Amount of trajectories: " + str(10**i) + " with time: " + str(func.timer(func.create_data,10, 10 , 10**i, 0.05)) )

    # plot_performances(test_functions,combined_results)
    #read_data_from_jsons = JSON_IO.read_data_from_jsons('results.txt')
    #greedy_expected = read_data_from_jsons['greedy']
    #assert greedy_expected == results
    #print('done')
    # print(nx.is_isomorphic(test1, test1_1))
    # r = create_results(test_functions, 5)

    # #loop through all optimal trajectories found, check if collision
    #  for name in test_functions:
    #      for result in r[name.__name__]:
    #          if func.check_for_collisions(result['trajectories']):
    #              print("error in " + name.__name__)

    # plot_performances(test_functions, r)