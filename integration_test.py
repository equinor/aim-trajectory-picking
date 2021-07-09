from numpy import result_type, true_divide
import aim_trajectory_picking.functions as func
import networkx as nx
import matplotlib.pyplot as plt
import JSON_IO 
import os
import datasets
import numpy as np
import random
import cProfile
from time import perf_counter
import math
import pandas as pd

algorithms = [func.greedy_algorithm, func.NN_algorithm,func.random_algorithm,
                     func.weight_transformation_algorithm, func.bipartite_matching_removed_collisions,
                     func.lonely_target_algorithm, func.reversed_greedy, func.invert_and_clique]
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

def plot_pandas_graph(algorithms, results, directory):
    results_dict = {}
    for algorithm in algorithms: 
        results_dict[algorithm.__name__ ] = 0

    dataset_names = [i for i in range(4)]
    # for filename in os.listdir(directory):
    #     if len(filename) < 10:
    #         break
    #     dataset_names.append(filename)
    pandas_dict = translate_results_to_panda_dict(results, algorithms)
    plotdata = pd.DataFrame(
        pandas_dict, 
        index=dataset_names
    )

    plotdata.plot(kind="bar", cmap =plt.get_cmap('Pastel1'))
    plt.title("Performance of Algorithms on Datasets")
    plt.xlabel("Dataset")
    plt.ylabel("Value")
    plt.show()

def translate_results_to_panda_dict(results, algorithms):
    pandas_dict = {}
    for algo in algorithms:
        name = algo.__name__
        pandas_dict[name] = [d['value'] for d in results[name]]
    return pandas_dict



def create_results(algorithms, no_of_datasets):
    combined_results = {}
    print("progress:")
    print('#'*1)
    for algorithm in algorithms:
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
            print("done with algorithm: " + algorithm.__name__ + " on dataset " + str(datasets.index(data)))

    for name in algorithms:
        for result in combined_results[name.__name__]:
            if func.check_for_collisions(result['trajectories']):
                print("error in " + name.__name__)

    return combined_results
    
def read_data_and_give_results():
    '''
    Description
    '''
    results = []
    directory = r'.\datasets'
    even_datasets = r'.\even_datasets'
    test_functions = [func.greedy_algorithm, func.NN_algorithm,func.random_algorithm,
                     func.weight_transformation_algorithm, func.bipartite_matching_removed_collisions,
                     func.lonely_target_algorithm, func.reversed_greedy, func.invert_and_clique]
    combined_results = {}
    for algorithm in test_functions:
        combined_results[algorithm.__name__] = []
    dataset_names = []
    iter = 0
    ITER_MAX = 2
    for filename in os.listdir(even_datasets):
        iter += 1
        if iter > ITER_MAX:
            break
        dataset_names.append(filename)
        fullpath = os.path.join(even_datasets,filename)
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
    for i in range(iter):
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

def runtime_test(list_of_algorithms,list_of_datasets_to_test):
    '''
    to test x algorithms on y datasets and save dataset size and runtime for each algorithm

    input list of algorithms and datasets
    
    Return either save result to a file  and return nothing or return as data time and dataset sizes for each algorithm so that another function can plot it nicely
    '''
    
if __name__ == '__main__':
    times = []
    max_iter = 4
    sets = []
    directory = r'.\datasets'
    # for filename in os.listdir(directory):
    #     if len(filename) < 10:
    #         break
    #     fullpath = os.path.join(directory,filename)
    #     sets.append(JSON_IO.read_trajectory_from_json(fullpath))
    for i in range(max_iter):
        _,_,traj = func.create_data(15,15,100)
        sets.append(traj)
        # start = perf_counter()
        # result = func.invert_and_clique(traj, False)
        # stop = perf_counter()
        # times.append(stop - start)
        # print("time for "+ str(i*10) + " trajectories: " + str(stop -start))
    # plt.figure()
    # plt.title("runtime analysis for clique")
    # x = [10*i for i in range(max_iter)]
    # plt.plot(x, times, label='runtime clique algo')
    # plt.plot(x, [e**2 /10**7 for e in x], label='quadratic', color='k')
    # plt.plot(x, [e**3 /10**7 for e in x], label='cubic', color='k')
    # plt.plot(x, [math.factorial(e) /(10**7) for e in x], label='factorial')
    # plt.legend()
    # plt.show()
    r = calculate_results(algorithms, sets)
    pandas_dict =  translate_results_to_panda_dict(r, algorithms)
  
    print(pandas_dict)
    plot_pandas_graph(algorithms, r, 'datasets')
    #plot_performances(algorithms,r)
