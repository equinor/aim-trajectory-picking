import argparse
from warnings import catch_warnings
from aim_trajectory_picking import algorithms as func
from aim_trajectory_picking import util
from aim_trajectory_picking import JSON_IO
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_datasets(dataset_folders, algorithms,refresh, filename='results.txt'):
    '''
    Function to find and/or create the given data and return it as a list.

    Parameters:
    dataset_folders: list<str>
    list of folders if which the datasets will be read and added to the list.

    algorithms: list<Function>
    list of algorithms which will be run on the given datasets. They exist to reduce runtime by not reading datasets which already have saved results

    refresh: bool
    bool to indicate if previously saved data will be ignored (if True) and results recalculated

    filename: str
    filename of the file to be read from

    Returns:
    --------
    data: list< Tuple( list<Trajectory>, list< Tuple(Trajectory, Trajectory)>)
    list of trajectories and their collisions

    dataset_names: list<str>
    list of dataset_names, either read (when reading from file) or None (when random data is chosen)
    '''
    data = []
    dataset_names = []
    if dataset_folders == None:
        print("None-type input file, bringing up runtime benchmarks")
        dataset_folders = []
        dataset_folders.append('testsets')
    try:
        if dataset_folders[0] == 'random':
            print("random data generation chosen")
            data = []
            num_donors = int(dataset_folders[1])
            num_targets = int(dataset_folders[2])
            num_trajectories = int(dataset_folders[3])
            collision_rate = float(dataset_folders[4])
            if len(dataset_folders) < 5:
                num_datasets = 1
            else:
                num_datasets = int(dataset_folders[5])
            for i in range(num_datasets):
                print("making dataset nr: " + str(i))
                _,_,trajectories, collisions = func.create_data(num_donors, num_targets, num_trajectories, collision_rate)
                data.append((trajectories, collisions))
                dataset_names.append('dataset_' + str(i)+ '.txt')
            return data, None
        elif dataset_folders[0] == 'increasing':
            upper_limit_trajectories = 10000
            print("increasing data generation chosen")
            data = []
            num_donors = int(dataset_folders[1])
            num_targets = int(dataset_folders[2])
            initial_num_trajectories = int(dataset_folders[3])
            collision_rate = float(dataset_folders[4])
            if len(dataset_folders) < 5:
                num_datasets = 1
            else:
                num_datasets = int(dataset_folders[5])
            for i in range(num_datasets):
                if initial_num_trajectories * (i+1) > upper_limit_trajectories:
                    break
                print("making dataset nr: " + str(i))
                _,_,trajectories, collisions = func.create_data(num_donors, num_targets, initial_num_trajectories * (i + 1), collision_rate)
                data.append((trajectories, collisions))
                dataset_names.append('dataset_' + str(i)+ '.txt')
            return data, None
        else:
            prev_results = get_previous_results(filename)
            datasets_as_string = ' '.join(map(str, dataset_folders))
            if len(os.listdir(datasets_as_string))==0:
                no_datasets = True
            else:
                no_datasets = False
                for folder in dataset_folders:
                    for filename in os.listdir(folder):
                        print("else file")
                        if refresh or not all(algo.__name__ in prev_results.keys() for algo in algorithms) or not all(filename in prev_results[algo.__name__].keys() for algo in algorithms):
                            fullpath = os.path.join(folder,filename)
                            data.append(JSON_IO.read_trajectory_from_json_v2(fullpath))
                            dataset_names.append(filename)
                            print(fullpath)
                        else:
                            dataset_names.append(filename)

    except Exception as e:
        print("exception thrown:", str(e))
        if len(data) == 0:
            print("Dataset arguments not recognized, reading from testsets instead.")
            for filename in os.listdir('testsets'):
                fullpath = os.path.join('testsets',filename)
                data.append(JSON_IO.read_trajectory_from_json_v2(fullpath))
                dataset_names.append(filename)
    return data, dataset_names, no_datasets

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i,y[i],y[i])

def plot_results_with_runtimes(algorithms, results, _dataset_names=0):
    '''
    Fully automatic function that plots the results per algorithm. 

    Important: the names of the algorithms must be keys in the results dictionary, and every value is a list \
        that consists of dictionaries, which again contain the value of trajectories of that specific algorithm on that \
            specific dataset.
    
    Parameters:
    -----------
    algorithms: List<Function>
        list of functions used to obtain results

    results: dictionary[algorithm.__name__][dataset] = {
                'value': int
                'trajectories: list<Trajectory>
                'runtime':float
    } 

    Returns:
    --------
    None
    '''
    means = []
    if _dataset_names == 0 or _dataset_names == None:
        dataset_names = [str(i) for i in range(len(results[algorithms[0].__name__]))]
    else:
        dataset_names = _dataset_names

    algo_names = [e.__name__ for e in algorithms]
    algo_runtimes = []

    if len(dataset_names) > 1:
        fig, axs = plt.subplots(2,1, figsize=(10,5))
        ax1 = axs[0]
        ax2 = axs[1]
        ax1.set_xlabel('Datasets')
        ax2.set_xlabel('Datasets')
        ax1.set_ylabel('Value')
        ax2.set_ylabel('Runtime (seconds)')
        ax1.title.set_text('Algorithm Performance')
        ax2.title.set_text('Algorithm Runtime')
        fig.tight_layout(pad=3)
        for algorithm in algorithms:
            results_per_dataset = [results[algorithm.__name__][dataset_name]['value'] for dataset_name in dataset_names]
            algo_runtimes =  [results[algorithm.__name__][dataset_name]['runtime'] for dataset_name in dataset_names]

            ax1.plot(dataset_names, results_per_dataset, label=algorithm.__name__)
            ax1.scatter(dataset_names, results_per_dataset, s=5, alpha=0.5) 
            ax2.plot(dataset_names, algo_runtimes, '--',label=algorithm.__name__)
            ax2.scatter(dataset_names, algo_runtimes, s=5, alpha=0.5)

            means.append(np.mean(results_per_dataset))
        leg1 = ax1.legend()
        leg1.set_draggable(state=True)
        # plt.xticks(rotation=45)
        leg2 = ax2.legend()
        leg2.set_draggable(state=True)
        plt.show()
    else:
        plt.figure()
        for algorithm in algorithms:
            results_per_dataset = [results[algorithm.__name__][dataset_name]['value'] for dataset_name in dataset_names]
            algo_runtimes =  [results[algorithm.__name__][dataset_name]['runtime'] for dataset_name in dataset_names]
            means.append(np.mean(results_per_dataset))
            plt.scatter(dataset_names, algo_runtimes, s=10, alpha=0.5)
        plt.xlabel('Algorithm Name')
        plt.ylabel('Runtime (seconds)')
        plt.title('Runtime graph')
        leg = plt.legend(algo_names)
        leg.set_draggable(state=True)
        plt.show()
    plt.figure(figsize=(12, 6))
    plt.bar(algo_names, means, color=(0.2, 0.4, 0.6, 0.6))
    addlabels(algo_names, means)
    for i, (name, height) in enumerate(zip(algo_names,  means)):
        plt.text(i, height/2, ' ' + name,
            ha='center', va='center', rotation=-90, fontsize=10)
    plt.xticks([])
    plt.title('Average Algorithm Performance')
    plt.xlabel('Algorithm Name')
    plt.ylabel('Average Value')
    plt.show()


def get_previous_results(filename):
    '''
    Function to read previous results from given file, if found.

    Parameters:
    -----------
    filename: str
    name of file previous results are located in

    Returns:
    prev_results: dictionary[algorithm.__name__][dataset] = {
                'value': int
                'trajectories: list<Trajectory>
                'runtime':float
    } 
    '''
    try:
        prev_results = JSON_IO.read_value_trajectories_runtime_from_file(filename)
    except:
        prev_results = {}
    return prev_results

def calculate_or_read_results(algos, _datasets,refresh, *, _is_random=False, filename='results.txt', _dataset_names=None):
    '''
    Function to either calculate the specified results or read them from file (if they have been calculated before)

    Parameters:
    algos: list<Function>
    list of functions to be ran on _datasets. Must accept list<Trajectory> and list<Tuple(Trajectory, Trajectory)> (collisions) and return\
        a result dictionary.

    _datasets: list< Tuple( list<Trajectory>, list< Tuple(Trajectory, Trajectory)>)
    list of trajectories and their collisions
    
    refresh: bool
    bool to indicate whether to recalculate results even if previously calculated.

    _is_random: bool
    bool to indicate if dataset is randomly generated or not, and therefore does not need to be saved to file

    filename: str
    file to read previous results from

    _dataset_names: list<str>
    list of dataset names, to be used in indexing dictionary. If random data generation is chosen, this will be None.

    Returns:
    --------
    results: dictionary[algorithm.__name__][dataset] = {
                'value': int
                'trajectories: list<Trajectory>
                'runtime':float
    } 
    '''

    dataset_names = [str(i) for i in range(len(_datasets))] if _dataset_names == None else _dataset_names

    prev_results = dict()
    if not _is_random:
        prev_results = get_previous_results(filename)

    for algorithm in algos:
        if algorithm.__name__ not in prev_results.keys():
            prev_results[algorithm.__name__] = {}

    for data in _datasets:
        data_name = dataset_names[_datasets.index(data)]
        for algorithm in algos:
            if not refresh and algorithm.__name__ in prev_results.keys() and _dataset_names!=None and data_name in prev_results[algorithm.__name__].keys():
                print("algorithm " + algorithm.__name__ + " on dataset " + data_name + " already in " + filename)
            else:
                #print(type(data))
                answer, runtime = func.timer(algorithm, data[0], data[1])
                answer['runtime'] = runtime
                prev_results[algorithm.__name__][data_name] = answer
                print("done with algorithm: " + algorithm.__name__ + " on dataset " + data_name)

    #check that trajectories are feasible
    for name in algos:
        for dataset in prev_results[name.__name__]:
            if func.check_for_collisions(prev_results[name.__name__][dataset]['trajectories']):
                print("error in algorithm" + name.__name__)

    if _dataset_names != None:
        JSON_IO.write_value_trajectories_runtime_from_file( prev_results, filename)
    return prev_results


def find_best_performing_algorithm(results, algorithms, used_datasets):
    best_result = 0
    algorithm_finder = 0
    best_algorithm_name_list = []
    matrix_list = []
    all_datasets_list =[]
    check_list = []

    for algorithm in algorithms:
        for all_datasets in results[algorithm.__name__]:
            if all_datasets not in all_datasets_list:
                all_datasets_list.append(all_datasets)

    used_datasets_set = set(used_datasets)
    all_datasets_set = set(all_datasets_list)
    intersection = used_datasets_set.intersection(all_datasets_set)
    intersection_as_list = sorted(list(intersection))

    for algorithm in algorithms:
    #     results_per_dataset = [results[algorithm.__name__][dataset_name]['value'] for dataset_name in results[algorithm.__name__]]   
    #     matrix_list.append(results_per_dataset)
        ram_list = []
        for element in intersection_as_list:
            if element in results[algorithm.__name__]:
                chosen_results_per_dataset = [results[algorithm.__name__][element]['value']]
                ram_list.append(chosen_results_per_dataset[0])

                if sum(chosen_results_per_dataset) > best_result:
                    best_result = sum(chosen_results_per_dataset)
                    for key in results.keys():
                        best_algorithm_name_list.append(key)
                    best_algorithm_name = best_algorithm_name_list[algorithm_finder]
        algorithm_finder += 1
        matrix_list.append(ram_list)
    map_matrix = list(map(max, zip(*matrix_list)))
    algorithm_finder_per_dataset = 0
    best_performing_algorithms = [[] for x in range(len(map_matrix))]
    for n in range(len(map_matrix)):
        for m in range(len(matrix_list)):
            if map_matrix[n] == matrix_list[m][n]:
                best_performing_algorithms[n].append(best_algorithm_name_list[m])
    print(best_performing_algorithms)
    for j in range(len(best_performing_algorithms)):
        listToStr = ' '.join(map(str, best_performing_algorithms[j]))
        print('On dataset', j+1, ',', listToStr, 'with value: ', map_matrix[j])
    print('Highest total value across all datasets: ', best_algorithm_name, ': value: ', best_result)


def translate_results_to_dict(results, algorithms):
    '''
    Translates the results to a dictionary to make plotting.

    Parameters:
    results
    algorithms
    '''
    results_as_dict = {}
    for algo in algorithms:
        name = algo.__name__
        results_as_dict[name] = [d['value'] for d in results[name]]
    return results_as_dict

def plot_algorithm_values_per_dataset(algorithms, results, directory): 
    results_dict = {}
    for algorithm in algorithms: 
        results_dict[algorithm.__name__ ] = 0

    dataset_names = [i for i in range(4)]
    pandas_dict = translate_results_to_dict(results, algorithms)
    plotdata = pd.DataFrame(
        pandas_dict, 
        index=dataset_names
    )

    plotdata.plot(kind="bar", cmap =plt.get_cmap('Pastel1'))
    plt.title("Performance of Algorithms on Datasets")
    plt.xlabel("Dataset")
    plt.ylabel("Value")
    plt.show()


    

def main():
    algorithms = {  'greedy' : func.greedy_algorithm, 
                'modified_greedy': func.modified_greedy,
                'NN' : func.NN_algorithm,
                # 'random' : func.random_algorithm,
                'weight_trans' :func.weight_transformation_algorithm, 
                # 'bipartite_matching' : func.bipartite_matching_removed_collisions,
                'lonely_target' : func.lonely_target_algorithm,
                'exact' : func.invert_and_clique,
                'ilp' : func.ILP,
                'cp-sat' : func.cp_sat_solver,
                # 'reversed_greedy_bipartite': func.reversed_greedy_bipartite_matching,
                # 'reversed_greedy_weight_trans' : func.reversed_greedy_weight_transformation,
                # 'reversed_greedy_regular_greedy' :func.reversed_greedy_regular_greedy,
                # 'bipartite_matching_v2': func.bip,
                #'approx_vertex_cover' :func.inverted_minimum_weighted_vertex_cover_algorithm # not working currently
                }
    not_runnable = [func.invert_and_clique]
    algo_choices = [ key for key in algorithms]
    algo_choices.append('all')
    algo_choices.append('runnable')

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=('''
            Trajectory picking algorithm for the AI for Maturation project
            Example of use:
            python run -datasets big_datasets -alg all
            python run -datasets random 15 15 1000 0.05 3 -alg greedy weight_trans bipartite''')
            ,epilog='This is the epilog',
            add_help=True)

    parser.add_argument('-alg',default='all',type=str,choices=algo_choices, nargs='*',help='Type of algorithm used',)
    parser.add_argument('-datasets',default='benchmark',nargs='*',type=str,help='String of the input data set folder, JSON format. \
        Default is datasets, and the algorithm will be run on datasets if the argument is not recognized. \
            Can also be random, with specified number of donors, targets and trajectories, in addition to collision rate and number of datasets\
                ex: random 10 10 100 0.05 10')
    parser.add_argument('-outputfile',metavar='Outputfile',type=str,default='optimal_trajectories.json',help='Filename string of output data result, JSON format')
    # could potentially add optional arguments for running test sets instead, or average of X trials
    parser.add_argument('-refresh', metavar='refresh', type = str, default='False', help='If true, ignores previous results and calculates the specified algorithms again')

    args = parser.parse_args()

    refresh = True if args.refresh == 'True' or args.refresh == 'true' else False
    if args.alg == 'all' or args.alg[0] == 'all':
        algos = [algorithms[key] for key in algorithms]
        if 'exact' not in args.alg:
            for unrunnable in not_runnable:
                algos.remove(unrunnable)
    else:
        algos = [algorithms[key] for key in args.alg]

    if 'benchmark' in args.datasets:
        results = JSON_IO.read_data_from_json_file('benchmark.txt')
        data_names = None
    else:
        data, data_names, empty_folder = get_datasets(args.datasets,algos,refresh)
        random_chosen = False
        if 'random' in args.datasets or 'increasing' in args.datasets: # Sets that would not have results saved from previous runs
            random_chosen = True   
    

        results = calculate_or_read_results(algos,data, refresh,_is_random=random_chosen, _dataset_names =data_names)
        find_best_performing_algorithm(results, algos)

    optimal_trajectory_dict = util.save_optimal_trajectories_to_file(results,args.outputfile,data_names)
    for dataset_name in optimal_trajectory_dict:
        print("Optimal trajectories for dataset ", dataset_name, ": ", optimal_trajectory_dict[dataset_name] )

        # Make a separate file for benchmark of algorithms
        # if 'increasing' in args.datasets:
        #     benchmark = results
        #     for key1 in benchmark:
        #         for key2 in benchmark[key1]:
        #             benchmark[key1][key2].pop("trajectories")
        #     JSON_IO.write_data_to_json_file('benchmark.txt',benchmark)


    random_chosen = False
    
    if args.datasets == None:
        random_chosen = False    
    elif 'random' in args.datasets:
        random_chosen = True
    results = calculate_or_read_results(algos,data,refresh, _is_random=random_chosen, _dataset_names =data_names)
    if empty_folder == False:
        find_best_performing_algorithm(results,algos,data_names)
        plot_results_with_runtimes(algos, results, data_names)

    else:
        print('No datasets found in datasetfolder')

if __name__ == '__main__':
    main()

