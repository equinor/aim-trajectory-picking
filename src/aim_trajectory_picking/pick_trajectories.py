import argparse
from warnings import catch_warnings
from aim_trajectory_picking import algorithms as func
from aim_trajectory_picking import util
from aim_trajectory_picking import JSON_IO
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from pathlib import Path
    

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

    parser.add_argument('-alg',default='cp-sat',type=str,choices=algo_choices, nargs='*',help='Type of algorithm used',)
    parser.add_argument('-datasets',default='benchmark',nargs='*',type=str,help='String of the input data set folder, JSON format. \
        Default is datasets, and the algorithm will be run on datasets if the argument is not recognized. \
            Can also be random, with specified number of donors, targets and trajectories, in addition to collision rate and number of datasets\
                ex: random 10 10 100 0.05 10')
    parser.add_argument('-outputfile',metavar='Outputfile',type=str,default='optimal_trajectories.json',help='Filename string of output data result, JSON format')
    # could potentially add optional arguments for running test sets instead, or average of X trials
    parser.add_argument('--refresh', '-r',   help='If given, ignores previous results and calculates the specified algorithms again',action='store_true')
    parser.add_argument('-show_figure',metavar='Show_figure',type=str,default='True',help='If True, do matplotlib.show to visualize runtime results')
    parser.add_argument('-save_benchmark',help='If given, save benchmark data to a benchmark.txt file',action='store_true')
    parser.add_argument('--verbose', '-v', help='Flag to indicate if the user wants the program to print more information', action='store_true')

    args = parser.parse_args()
    arguments = sys.argv[1:]
    refresh = args.refresh
    if args.alg == 'all' or args.alg[0] == 'all' or len(arguments) == 0:
        algos = [algorithms[key] for key in algorithms]
        if 'exact' not in args.alg:
            for unrunnable in not_runnable:
                algos.remove(unrunnable)
    elif isinstance(args.alg, list):
        algos = [algorithms[key] for key in args.alg]
    else:
        algos = [algorithms[args.alg]]

    if 'benchmark' in args.datasets:
        p = Path("benchmark.txt").resolve()
        results = JSON_IO.read_data_from_json_file(str(p))
        data_names = None
        util.plot_results_with_runtimes(algos, results, data_names,show_figure=args.show_figure)
    else:
        if 'random' in args.datasets or 'increasing' in args.datasets: # Sets that would not have results saved from previous runs
            random_chosen = True
            data,data_names = util.get_datasets(args.datasets,algos,refresh)
            results = util.calculate_or_read_results(algos,data, refresh,_is_random=random_chosen, _dataset_names =data_names,verbose=args.verbose)   
            util.plot_results_with_runtimes(algos, results, data_names,show_figure=args.show_figure)
        else:
            data, data_names, empty_folder = util.get_datasets(args.datasets,algos,refresh)
            random_chosen = False
            results = util.calculate_or_read_results(algos,data, refresh,_is_random=random_chosen, _dataset_names =data_names,verbose=args.verbose)
            list_of_used_datasets, best_algorithms_per_dataset, highest_value_per_dataset, best_algorithm_name, best_result = util.find_best_performing_algorithm(results,algos,data_names)
            
            if empty_folder == False:
                util.find_best_performing_algorithm(results,algos,data_names)
                util.plot_results_with_runtimes(algos, results, data_names,show_figure=args.show_figure)
            else:
                print('No datasets found in datasetfolder')
            
            if args.verbose:
                for j in range(len(list_of_used_datasets)):
                    print('On dataset: ', list_of_used_datasets[j], ',', best_algorithms_per_dataset[j], 'with value: ', highest_value_per_dataset[j])
                print('Highest total value across all datasets: ', best_algorithm_name, ': value: ', best_result)

            optimal_trajectory_dict = util.save_optimal_trajectories_to_file(results,args.outputfile,data_names)
            for i in range(len(list_of_used_datasets)):
                dataset_name = list_of_used_datasets[i]
                print("Optimal trajectories for dataset ", dataset_name, ": ", optimal_trajectory_dict[dataset_name])



        # Make a separate file for benchmark of algorithms
        if 'increasing' in args.datasets and args.save_benchmark == True:
            benchmark = results
            for key1 in benchmark:
                for key2 in benchmark[key1]:
                    benchmark[key1][key2].pop("trajectories")
            JSON_IO.write_data_to_json_file('benchmark.txt',benchmark)
            print(" ---------- saved benchmark ----------")

if __name__ == '__main__':
    main()

