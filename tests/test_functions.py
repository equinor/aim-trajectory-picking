import pytest
import aim_trajectory_picking.functions as func
import JSON_IO

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
    donors, targets, trajectories = func.create_data(3,3,10,0.2)
    filename = 'JSON_test.txt'
    JSON_IO.write_data_to_json_file(filename, trajectories)
    read_trajectories = JSON_IO.read_data_from_json_file(filename)
    for i in range(len(read_trajectories)):
        print(trajectories[i])
        print(read_trajectories[i])
    print(trajectories[0] == read_trajectories[0])
    assert all([a == b for a, b in zip(trajectories, read_trajectories)])
