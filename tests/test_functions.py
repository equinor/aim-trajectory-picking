import pytest
import src.aim_trajectory_picking.functions as func
import src.aim_trajectory_picking.JSON_IO as JSON_IO
import src.aim_trajectory_picking.integration_testing as int_test
import os

def algorithm_test_function(algorithm):
    targeted_result_list = []
    directory = r'.\datasets'
    test_functions = [algorithm]
    combined_results = {}
    for algorithm in test_functions:
        combined_results[algorithm.__name__] = []
    dataset_names = []
    for filename in os.listdir(directory):
        dataset_names.append(filename)
        fullpath = os.path.join(directory,filename)
        dataset1_after = JSON_IO.read_trajectory_from_json(fullpath)

        for algorithm in test_functions:
            answer = algorithm(dataset1_after, False)
            combined_results[algorithm.__name__].append(answer)

    for i in range(5):
        targeted_result = combined_results[algorithm.__name__][i]['value']
        targeted_result_list.append(targeted_result)
    return targeted_result_list

# Test the mutually exclusive function on three different arbitrary sets
@pytest.mark.parametrize("test_input1,test_input2,test_input3,test_input4,expected", [("T1","D1","T1","D2",True), ("T1","D1","T2","D1",True), ("T1","D1","T2","D2",False)])
def test_mutually_exclusive_trajectories(test_input1,test_input2,test_input3,test_input4, expected):
    t1 = func.Trajectory(0, test_input1, test_input2, 5)
    t2 = func.Trajectory(1, test_input3, test_input4, 5)
    assert func.mutually_exclusive_trajectories(t1, t2) == expected

# Test if a collision is correctly added and detected
def test_add_collision():
    t1 = func.Trajectory(2, 'D1', 'T1', 5)
    t2 = func.Trajectory(2, 'D2', 'T2', 5)
    t1.add_collision(t2)
    assert func.mutually_exclusive_trajectories(t2,t1) == True

# Test whether the convertion of data to and from JSON-format results in the same result
# NB: the documentation does not garantee the result to be equal if there are non-string keys
def test_JSON_IO():
    _, _, trajectories = func.create_data(3,3,10,0.2)
    filename = 'JSON_test.txt'
    JSON_IO.write_trajectory_to_json(filename, trajectories)
    read_trajectories = JSON_IO.read_trajectory_from_json(filename)
    for i in range(len(read_trajectories)):
        print(trajectories[i])
        print(read_trajectories[i])
    print(trajectories[0] == read_trajectories[0])
    assert all([a == b for a, b in zip(trajectories, read_trajectories)])

def test_greedy_on_datasets_0_to_4():
    targeted_result_list = algorithm_test_function(func.greedy_algorithm)
    assert targeted_result_list == [32, 20, 26, 31, 23]

def test_NN_on_datasets_0_to_4():
    targeted_result_list = algorithm_test_function(func.NN_algorithm)
    assert targeted_result_list == [32, 20, 26, 32, 24]

def test_weight_on_datasets_0_to_4():
    targeted_result_list = algorithm_test_function(func.weight_transformation_algorithm)
    assert targeted_result_list == [29, 24, 26, 31, 24]

def test_bipartite_removed_collision_on_datasets_0_to_4():
    targeted_result_list = algorithm_test_function(func.bipartite_matching_removed_collisions)
    assert targeted_result_list == [32, 24, 26, 32, 23]

def test_lonely_target_on_datasets_0_to_4():
    targeted_result_list = algorithm_test_function(func.lonely_target_algorithm)
    assert targeted_result_list == [32, 11, 23, 32, 24]

def test_reverse_greedy_on_datasets_0_to_4():
    targeted_result_list = algorithm_test_function(func.reversed_greedy)
    assert targeted_result_list == [29, 24, 26, 31, 23]