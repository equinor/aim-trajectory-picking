import argparse
from warnings import catch_warnings
from aim_trajectory_picking import functions as func
from aim_trajectory_picking import JSON_IO
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_datasets(dataset_folders):
    '''
    Function to find and/or create the given data and return it as a list.

    Parameters:
    dataset_folders:list
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
                _,_,trajectories = func.create_data(num_donors, num_targets, num_trajectories, collision_rate)
                data.append(trajectories)
                dataset_names.append('dataset_' + str(i)+ '.txt')
            return data, None
        else:
            for folder in dataset_folders:
                print("using data from folder")
                for filename in os.listdir(folder):
                    fullpath = os.path.join(folder,filename)
                    data.append(JSON_IO.read_trajectory_from_json(fullpath))
                    dataset_names.append(filename)
    except:
        pass
    if len(data) == 0:
        print("Dataset arguments not recognized, reading from testsets folder instead.")
        for filename in os.listdir('testsets'):
            fullpath = os.path.join('testsets',filename)
            data.append(JSON_IO.read_trajectory_from_json(fullpath))
            dataset_names.append(filename)
    return data, dataset_names


def plot_results_with_runtimes(algorithms, results,_dataset_names=0):
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
    fig, axs = plt.subplots(2,1)
    means = []
    if _dataset_names == 0 or _dataset_names == None:
        dataset_names = [str(i) for i in range(len(results[algorithms[0].__name__]))]
    else:
        dataset_names = _dataset_names

    algo_names = [e.__name__ for e in algorithms]
    algo_runtimes = []
    for algorithm in algorithms:
        results_per_dataset = [results[algorithm.__name__][dataset_name]['value'] for dataset_name in results[algorithm.__name__]]
        algo_runtimes =  [results[algorithm.__name__][dataset_name]['runtime'] for dataset_name in results[algorithm.__name__]]
        axs[0].plot(dataset_names, results_per_dataset, label=algorithm.__name__) 
        axs[1].plot(dataset_names, algo_runtimes, '--',label=algorithm.__name__)
        means.append(np.mean(results_per_dataset))
    axs[1].plot(dataset_names, [x**2 for x in range(len(dataset_names))],'k', label='n^2')
    axs[1].plot(dataset_names, [x for x in range(len(dataset_names))],'b', label='n')
    axs[0].legend()
    plt.xticks(rotation=45)
    axs[1].legend()
    plt.show()
    plt.figure()
    plt.bar(algo_names, means)
    plt.xticks(rotation=45)
    plt.show()

def get_previous_results(filename):
    try:
        prev_results = JSON_IO.read_value_trajectories_runtime_from_file(filename)
    except:
        prev_results = {}
    return prev_results

def calculate_or_read_results(algos, _datasets, *, _is_random=False, filename='results.txt', _dataset_names=None):

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
            if algorithm.__name__ in prev_results.keys() and _dataset_names!=None and data_name in prev_results[algorithm.__name__].keys():
                print("algorithm " + algorithm.__name__ + " on dataset " + data_name + " already in " + filename)
            else:
                answer, runtime = func.timer(algorithm, data)
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

def translate_results_to_pandas_dict(results, algorithms):
    pandas_dict = {}
    for algo in algorithms:
        name = algo.__name__
        pandas_dict[name] = [d['value'] for d in results[name]]
    return pandas_dict

def plot_algorithm_values_per_dataset(algorithms, results, directory): 
    results_dict = {}
    for algorithm in algorithms: 
        results_dict[algorithm.__name__ ] = 0

    dataset_names = [i for i in range(4)]
    pandas_dict = translate_results_to_pandas_dict(results, algorithms)
    plotdata = pd.DataFrame(
        pandas_dict, 
        index=dataset_names
    )

    plotdata.plot(kind="bar", cmap =plt.get_cmap('Pastel1'))
    plt.title("Performance of Algorithms on Datasets")
    plt.xlabel("Dataset")
    plt.ylabel("Value")
    plt.show()


    


if __name__ == '__main__':
    algorithms = {  'greedy' : func.greedy_algorithm, 
                'NN' : func.NN_algorithm,
                'random' : func.random_algorithm,
                'weight_trans' :func.weight_transformation_algorithm, 
                'bipartite_matching' : func.bipartite_matching_removed_collisions,
                'lonely_target' : func.lonely_target_algorithm,
                'exact' : func.invert_and_clique,
                'reversed_greedy_bipartite': func.reversed_greedy_bipartite_matching,
                'reversed_greedy_weight_trans' : func.reversed_greedy_weight_transformation,
                'reversed_greedy_regular_greedy' :func.reversed_greedy_regular_greedy,
                # 'bipartite_matching_v2': func.bip,
                # 'approx_vertex_cover' :func.minimum_weighted_vertex_cover_algorithm # not working currently
                }
    not_runnable = [func.invert_and_clique]
    algo_choices = [ key for key in algorithms]
    algo_choices.append('all')
    algo_choices.append('runnable')

    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
        description=('''\
            Trajectory picking algorithm for the AI for Maturation project
            Example of use:
            python pick_trajectories -datasets big_datasets -alg all 
            python pick_trajectories -datasets random 15 15 1000 0.05 3 -alg greedy weight_trans bipartite
            --------------------------------------------------------------
            JSON inputfile format:
            {
                "trajectories": [
                    {
                    "id": str,
                    "donor": str,
                    "target": str,
                    "value": int,
                    "collisions": [
                        id, ...
                        ]
                    },
                    ...    
                ]
            }''')
            ,epilog='This is the epilog',
            add_help=True)

    parser.add_argument('-alg',default='all',type=str,choices=algo_choices, nargs='*',help='Type of algorithm used (default: greedy)',)
    parser.add_argument('-datasets',metavar='Datasets',nargs='*',type=str,help='String of the input data set folder, JSON format. \
        Default is datasets, and the algorithm will be run on datasets if the argument is not recognized. \
            Can also be random, with specified number of donors, targets and trajectories, in addition to collision rate and number of datasets\
                ex: random 10 10 100 0.05 10')
    parser.add_argument('-outputfile',metavar='Outputfile',type=str,default='trajectories.txt',help='Filename string of output data result, JSON format')
    # could potentially add optional arguments for running test sets instead, or average of X trials

    args = parser.parse_args()
    data, data_names = get_datasets(args.datasets)

    if args.alg or args.alg[0] == 'all':
        algos = [algorithms[key] for key in algorithms]
        for unrunnable in not_runnable:
            algos.remove(unrunnable)
    elif args.alg or args.alg[0] == 'all' and args.alg[1] == 'exact':
        algos = [algorithms[key] for key in algorithms]
    else:
        algos = [algorithms[key] for key in args.alg]

    random_chosen = False
    
    # To remove the default datasets from th erandom choice when using random datasets
    if args.datasets == None:
        random_chosen = False    
    elif 'random' in args.datasets:
        random_chosen = True

    results = calculate_or_read_results(algos,data, _is_random=random_chosen, _dataset_names =data_names)
    plot_results_with_runtimes(algos, results, data_names)

'''
 input <- from user
 parse input
 results = (algorithm, dataset):
    if result not found
        read/calculate dataset
        save result
    else
        import result from file
plot results
'''