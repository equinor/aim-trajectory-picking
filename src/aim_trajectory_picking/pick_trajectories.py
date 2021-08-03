import argparse
from warnings import catch_warnings
from aim_trajectory_picking import algorithms as func
from aim_trajectory_picking import util
from aim_trajectory_picking import JSON_IO
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
    

def main():
    algorithms = {
                'greedy' : func.greedy_algorithm, 
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
        util.plot_results_with_runtimes(algos, results, data_names)
    else:
        data, data_names, empty_folder = util.get_datasets(args.datasets,algos,refresh)
        random_chosen = False
        if 'random' in args.datasets or 'increasing' in args.datasets: # Sets that would not have results saved from previous runs
            random_chosen = True   
    

        results = util.calculate_or_read_results(algos,data, refresh,_is_random=random_chosen, _dataset_names =data_names)
        if empty_folder == False:
            util.find_best_performing_algorithm(results,algos,data_names)
            util.plot_results_with_runtimes(algos, results, data_names)
        else:
            print('No datasets found in datasetfolder')

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

if __name__ == '__main__':
    main()

