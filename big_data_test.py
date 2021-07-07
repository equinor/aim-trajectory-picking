from matplotlib.pyplot import plot
import aim_trajectory_picking.functions as func
from integration_test import plot_performances
import JSON_IO

if __name__ == '__main__':
    test_functions = [func.greedy_algorithm, func.NN_algorithm,func.random_algorithm, func.weight_transformation_algorithm, func.bipartite_matching_removed_collisions, func.lonely_target_algorithm]
    combined_results = {}
    for algorithm in test_functions:
        combined_results[algorithm.__name__] = []
    BIG_data = JSON_IO.read_trajectory_from_json('big_datasets/highD_highT_highT.txt')
    print("done reading")


    for algorithm in test_functions:
            answer = algorithm(BIG_data, False)
            combined_results[algorithm.__name__].append(answer)
            print("done with algorithm: " + algorithm )
    
    JSON_IO.write_data_to_json_file('big_data_results.txt', combined_results)
    for i in range(5):
        for algorithm in test_functions:
            print(algorithm.__name__ +  " gave result: " + str(combined_results[algorithm.__name__][i]['value']))
    plot_performances(test_functions, combined_results)
    
        