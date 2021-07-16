import argparse
from warnings import catch_warnings
from aim_trajectory_picking.integration_testing import plot_performances
import functions as func
import os
import JSON_IO
import igraph
import matplotlib.pyplot as plt
import numpy as np

# TODO set defaults visable, clean up names, other?
# TODO save and read results, no arguments run = runtimes of algorithms, specify refresh data or not

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
        dataset_folders.append('datasets')
    # for i in range(len(dataset_folders)):
    try:
        if dataset_folders[0] == 'random':
            print("random data generation chosen")
            data = []
            no_donors = int(dataset_folders[1])
            no_targets = int(dataset_folders[2])
            no_trajectories = int(dataset_folders[3])
            collision_rate = float(dataset_folders[4])
            no_datasets = int(dataset_folders[5])
            for i in range(no_datasets):
                print("making dataset nr: " + str(i))
                _,_,tra = func.create_data(no_donors, no_targets, no_trajectories, collision_rate)
                data.append(tra)
                dataset_names.append('dataset_' + str(i)+ '.txt')
            return data, None
        else:
            for folder in dataset_folders:
                for filename in os.listdir(folder):
                    print("else file")
                    fullpath = os.path.join(folder,filename)
                    data.append(JSON_IO.read_trajectory_from_json(fullpath))
                    dataset_names.append(filename)
    except:
        pass
    if len(data) == 0:
        print("Dataset arguments not recognized, reading from datasets instead.")
        for filename in os.listdir('datasets'):
            fullpath = os.path.join('datasets',filename)
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
    #plt.figure(figsize=(9,3))
    fig, axs = plt.subplots(2,1)
    #fig.title('Performance of various algorithms on trajectory problem')
    means = []

    if _dataset_names == 0 or _dataset_names == None:
        dataset_names = [str(i) for i in range(len(results[algorithms[0].__name__]))]
    else:
        dataset_names = _dataset_names
    #axs2 = axs.twinx()

    algo_names = [e.__name__ for e in algorithms]
    algo_runtimes = []
    for algorithm in algorithms:
        results_per_dataset = [results[algorithm.__name__][item]['value'] for item in results[algorithm.__name__]]
        algo_runtimes =  [results[algorithm.__name__][item]['runtime'] for item in results[algorithm.__name__]]
        print(results_per_dataset)
        #print(algo_runtimes)
        #
        axs[0].plot(dataset_names, results_per_dataset, label=algorithm.__name__) 
        #plt.subplot(221)
        axs[1].plot(dataset_names, algo_runtimes, '--',label=algorithm.__name__)
        means.append(np.mean(results_per_dataset))
    axs[1].plot(dataset_names, [x**2 for x in range(len(dataset_names))],'k', label='n^2')
    axs[1].plot(dataset_names, [x for x in range(len(dataset_names))],'b', label='n')
    axs[1].plot(dataset_names, [x*np.log(x) for x in range(len(dataset_names))],'g', label='n log n')
    axs[0].legend()
    plt.xticks(rotation=45)
    axs[1].legend()
    #axs[1].xticks(rotation=45)
    #axs[0,0].xticks(rotation=45)
    #plt.subplot(122)
    #axs[0,1].bar(algo_names, means)
    #axs[0,1].xticks(rotation=45)
    #fig.tight_layout()
    plt.show()
    plt.figure()
    plt.bar(algo_names, means)
    plt.xticks(rotation=45)
    plt.show()

def calculate_or_read_results(algos, _datasets, *, filename='results.txt', _dataset_names=None):
    combined_results = {}
    if _dataset_names == None:
        dataset_names = [str(i) for i in range(len(_datasets))]
    else:
        dataset_names = _dataset_names

    try:
        prev_results = JSON_IO.read_value_trajectories_runtime_from_file(filename)
        for algo in algos:
            if algo.__name__ not in prev_results.keys():
                prev_results[algo.__name__] = {}
    except:
        prev_results = {}
        for algorithm in algos:
            prev_results[algorithm.__name__] = {}

    for algorithm in algos:
        combined_results[algorithm.__name__] = {}

    for data in _datasets:
        data_name = dataset_names[_datasets.index(data)]
        for algorithm in algos:
            #print(prev_results[algorithm.__name__].keys())
            try:
                if _dataset_names != None and data_name in prev_results[algorithm.__name__].keys():
                    combined_results[algorithm.__name__][data_name] = prev_results[algorithm.__name__][data_name]
                    print("algorithm " + algorithm.__name__ + " on dataset " + data_name + " already in " + filename)
                else:
                    answer, runtime = func.timer(algorithm, data)
                    #print("answer:")
                    #print(answer)
                    answer['runtime'] = runtime
                    combined_results[algorithm.__name__][data_name] = answer
                    prev_results[algorithm.__name__][data_name] = answer
                    print("done with algorithm: " + algorithm.__name__ + " on dataset " + data_name)
            except:
                answer, runtime = func.timer(algorithm, data, False)
                #print("answer:")
                #print(answer)
                answer['runtime'] = runtime
                combined_results[algorithm.__name__][data_name] = answer
                prev_results[algorithm.__name__][data_name] = answer
                print("done with algorithm: " + algorithm.__name__ + " on dataset " + data_name)

    for name in algos:
        for key in combined_results[name.__name__]:
            if func.check_for_collisions(combined_results[name.__name__][key]['trajectories']):
                print("error in algorithm" + name.__name__)

    if _dataset_names != None:
        JSON_IO.write_value_trajectories_runtime_from_file( prev_results, filename)
    return combined_results
    


if __name__ == '__main__':
    algorithms = {  'greedy' : func.greedy_algorithm, 
                'NN' : func.NN_algorithm,
                #'random' : func.random_algorithm,
                'weight_trans' :func.weight_transformation_algorithm, 
                'bipartite_matching' : func.bipartite_matching_removed_collisions,
                'lonely_target' : func.lonely_target_algorithm,
                'exact' : func.invert_and_clique,
                # 'reversed_greedy_bipartite': func.reversed_greedy_bipartite_matching,
                # 'reversed_greedy_weight_trans' : func.reversed_greedy_weight_transformation,
                # 'reversed_greedy_regular_greedy' :func.reversed_greedy_regular_greedy,
                #'bipartite_matching_v2': func.bip
                #'approx_vertex_cover' :func.inverted_minimum_weighted_vertex_cover_algorithm # not working currently
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
    
    # print(args)
    # print(args.datasets)
    # print(args.outputfile)
    # print(args.alg)    
    
    data, data_names = get_datasets(args.datasets)

    g = igraph.Graph()
    if args.alg == 'all':
        algos = [algorithms[key] for key in algorithms]
    elif args.alg[0] == 'all' and args.alg[1] == 'runnable':
        algos = [algorithms[key] for key in algorithms]
        for unrunnable in not_runnable:
            algos.remove(unrunnable)
    else:
        algos = [algorithms[key] for key in args.alg]

    results = calculate_or_read_results(algos,data, _dataset_names =data_names)        

    plot_results_with_runtimes(algos, results, data_names)

