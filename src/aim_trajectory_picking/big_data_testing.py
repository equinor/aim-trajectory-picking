from src.aim_trajectory_picking.integration_testing import plot_performances
from src.aim_trajectory_picking import functions as func
from src.aim_trajectory_picking import JSON_IO
from matplotlib.pyplot import plot
import time

if __name__ == '__main__':
    #test_functions = [func.greedy_algorithm, func.NN_algorithm,func.random_algorithm, func.weight_transformation_algorithm, func.bipartite_matching_removed_collisions, func.lonely_target_algorithm]
    test_functions = [func.modified_greedy]
    combined_results = {}
    for algorithm in test_functions:
        combined_results[algorithm.__name__] = []
    start = time.perf_counter()
    BIG_data, collisions = JSON_IO.read_trajectory_from_json_v2('big_datasets/highD_highT_highT.txt')
    stop = time.perf_counter()
    print("done reading with time: " + str(start-stop))


    for algorithm in test_functions:
            answer = algorithm(BIG_data,collisions, False)
            combined_results[algorithm.__name__].append(answer)
            print("done with algorithm: " + algorithm.__name__ )
    
    JSON_IO.write_data_to_json_file('big_data_results.txt', combined_results)
    for i in range(5):
        for algorithm in test_functions:
            print(algorithm.__name__ +  " gave result: " + str(combined_results[algorithm.__name__][i]['value']))
    plot_performances(test_functions, combined_results)
    
        