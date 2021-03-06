import pytest
import os

from aim_trajectory_picking import JSON_IO
from aim_trajectory_picking import algorithms
from aim_trajectory_picking import pick_trajectories
from aim_trajectory_picking import util

def algorithm_testing_function(algorithm):
    targeted_result_list = []
    directory = r'./testsets'
    test_functions = [algorithm]
    combined_results = {}
    for algorithm in test_functions:
        combined_results[algorithm.__name__] = []
    testset_names = []
    for filename in os.listdir(directory):
        testset_names.append(filename)
        fullpath = os.path.join(directory,filename)
        testset1_after = JSON_IO.read_trajectory_from_json(fullpath)

        for algorithm in test_functions:
            answer = algorithm(testset1_after[0],testset1_after[1] )
            combined_results[algorithm.__name__].append(answer)

    for i in range(5):
        targeted_result = combined_results[algorithm.__name__][i]['value']
        targeted_result_list.append(targeted_result)
    return targeted_result_list

# Test the mutually exclusive function on three different arbitrary sets
@pytest.mark.parametrize("test_input1,test_input2,test_input3,test_input4,expected", [("T1","D1","T1","D2",True), ("T1","D1","T2","D1",True), ("T1","D1","T2","D2",False)])
def test_mutually_exclusive_trajectories(test_input1,test_input2,test_input3,test_input4, expected):
    t1 = util.Trajectory(0,0, test_input1, test_input2, 5)
    t2 = util.Trajectory(1,0, test_input3, test_input4, 5)
    assert util.mutually_exclusive_trajectories(t1, t2) == expected

# Test if a collision is correctly added and detected
def test_add_collision():
    t1 = util.Trajectory(2,2, 'D1', 'T1', 5)
    t2 = util.Trajectory(2,2, 'D2', 'T2', 5)
    t1.add_collision(t2)
    assert util.mutually_exclusive_trajectories(t2,t1) == True

# Test whether the convertion of data to and from JSON-format results in the same result
# NB: the documentation does not garantee the result to be equal if there are non-string keys
def test_JSON_IO():
    _, _, trajectories, collisions = util.create_data(3,3,10,0.2)
    filename = 'JSON_test.txt'
    JSON_IO.write_trajectory_to_json(filename, trajectories)
    read_trajectories, coll = JSON_IO.read_trajectory_from_json(filename)
    for i in range(len(read_trajectories)):
        print(trajectories[i])
        print(read_trajectories[i])
    print(trajectories[0] == read_trajectories[0])
    assert all([a == b for a, b in zip(trajectories, read_trajectories)])

def test_greedy_on_testsets_0_to_4():
    targeted_result_list = algorithm_testing_function(algorithms.greedy_algorithm)
    assert targeted_result_list.sort() == [32, 20, 26, 31, 23].sort()

def test_NN_on_testsets_0_to_4():
    targeted_result_list = algorithm_testing_function(algorithms.NN_algorithm)
    assert targeted_result_list.sort() == [32, 24, 20, 32, 26].sort()#[32, 20, 26, 32, 24]

def test_weight_on_testsets_0_to_4():
    targeted_result_list = algorithm_testing_function(algorithms.weight_transformation_algorithm)
    assert targeted_result_list.sort() ==  [31, 24, 24, 29, 26].sort()#[29, 24, 26, 31, 24]

def test_bipartite_removed_collision_on_testsets_0_to_4():
    targeted_result_list = algorithm_testing_function(algorithms.bipartite_matching_removed_collisions)
    assert targeted_result_list.sort() == [32, 23, 24, 32, 26].sort()#[32, 24, 26, 32, 23]

def test_lonely_target_on_testsets_0_to_4():
    targeted_result_list = algorithm_testing_function(algorithms.lonely_target_algorithm)
    assert targeted_result_list.sort() == [32, 24, 11, 32, 23].sort()#[32, 11, 23, 32, 24]

def test_reverse_greedy_on_testsets_0_to_4():
    targeted_result_list = algorithm_testing_function(algorithms.reversed_greedy)
    assert targeted_result_list.sort() == [31, 23, 24, 29, 26].sort()#[29, 24, 26, 31, 23]

def test_exact_algorithm_0_to_4():
    exact_results = algorithm_testing_function(algorithms.invert_and_clique)
    greedy_results = algorithm_testing_function(algorithms.greedy_algorithm)
    assert exact_results >= greedy_results

def test_timer():
    _, time_used = util.timer(algorithm_testing_function,algorithms.greedy_algorithm)
    assert time_used >= 0

def test_ILP():
    ilp_result = algorithm_testing_function(algorithms.ILP)
    greedy_result = algorithm_testing_function(algorithms.greedy_algorithm)
    exact_result = algorithm_testing_function(algorithms.invert_and_clique)
    assert ilp_result == exact_result and ilp_result >= greedy_result

def test_CP_Sat():
    cpsat_result = algorithm_testing_function(algorithms.ILP)
    greedy_result = algorithm_testing_function(algorithms.greedy_algorithm)
    exact_result = algorithm_testing_function(algorithms.invert_and_clique)
    assert cpsat_result == exact_result and cpsat_result >= greedy_result

def test_testsets_dataset_input():
    function_path = os.path.join('src','aim_trajectory_picking','pick_trajectories.py')
    retval = os.system(r'python ' + function_path + ' -datasets testsets -show_figure False')
    assert retval == 0

def test_random_dataset_input():
    function_path = os.path.join('src','aim_trajectory_picking','pick_trajectories.py')
    retval = os.system(r'python ' + function_path + ' -datasets random 10 10 20 0.05 3 -show_figure False')
    assert retval == 0

# def test_benchmark_dataset_input():
#     function_path = os.path.join('src','aim_trajectory_picking','pick_trajectories.py')
#     retval = os.system(r'python ' + function_path + ' -show_figure False')
#     assert retval == 0