import json
import aim_trajectory_picking.demo_uke_1 as dem

donors, targets, trajectories = dem.create_data(2,2,4,0.3)
print(donors)
print(targets)
print(trajectories)

JSON_trajectories = {}
JSON_trajectories['trajectories'] = []
for x in range(3):
    trajectory = {}
    trajectory['id'] = trajectories[x].id
    trajectory['donor'] = trajectories[x].donor
    trajectory['target'] = trajectories[x].target
    trajectory['collisions'] = trajectories[x].collisions
    JSON_trajectories['trajectories'].append(trajectory)
    

print(JSON_trajectories)

with open('JSON_test_set.txt', 'w') as outfile: 
    json.dump(JSON_trajectories, outfile, sort_keys=False, indent=4)


file = open('JSON_test_set.txt','r')

input_data = json.loads(file.read())
print(input_data)