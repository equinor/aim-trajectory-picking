import JSON_IO as jio
import time
import os
import aim_trajectory_picking.functions as func
import matplotlib.pyplot as plt

'''
Might be unneeded now
'''




# start = time.perf_counter()
# jio.generate_increasing_datasets(5,10)
# stop = time.perf_counter()
# diff = stop-start
# print("time to generate: "+ str(diff))

visualize = True

# Read dataset
directory = r'.\timesets'
time_taken = []
values = []

for i in range(4):
    filename = 'increasing_set_'+str(i+1)+'.json'
    fullpath = os.path.join(directory,filename)
    nodes_list = jio.read_trajectory_from_json(fullpath)
    G = func.transform_graph(nodes_list)

    Start_time = time.perf_counter()
    opt_trajectory = func.greedy_algorithm(nodes_list)
    Stop_time = time.perf_counter()
    time_taken.append(Stop_time-Start_time)
    
    # value = 0
    # for e in opt_trajectory:
    #     value = value + e.value
    # values.append(value)

print(time_taken)