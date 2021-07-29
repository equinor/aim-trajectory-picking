from ortools.sat.python import cp_model
import os
from aim_trajectory_picking import functions as func
from aim_trajectory_picking import JSON_IO


def get_donor_and_target_collisions(trajectories):
    donor_dict = {}
    target_dict = {}
    for t in trajectories:
        if t.donor in donor_dict:
            donor_dict[t.donor].append(t)
        else:
            donor_dict[t.donor] = [t]
        if t.target in target_dict:
            target_dict[t.target].append(t)
        else:
            target_dict[t.target] = [t]
    return donor_dict, target_dict

def cp_sat_solver(trajectories, collisions):
    model = cp_model.CpModel()
    num_trajectories = len(trajectories)
    donor_dict, target_dict = get_donor_and_target_collisions(trajectories)
    x = {}
    for i in range(num_trajectories):
        x[i] = model.NewIntVar(0,1,str(i))
    
    for (t1, t2) in collisions:
        model.Add(x[t1.id] + x[t2.id] <=1)
    for donor in donor_dict:
        model.Add(sum([x[t.id] for t in donor_dict[donor]]) <= 1)
    for target in target_dict:
        model.Add(sum([x[t.id] for t in target_dict[target]]) <= 1)
    
    model.Maximize(sum([x[i] * trajectories[i].value for i in range(num_trajectories)]))

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    # if status == cp_model.OPTIMAL:
    #     print('optimal solution found')
    # else:
    #     print(status)
    
    #print('Objective value:', solver.ObjectiveValue)
    optimal_trajectories = []
    for i in range(len(x)):
        if solver.Value(x[i]) == 1:
            optimal_trajectories.append(trajectories[i])
    print(sum(t.value for t in optimal_trajectories))
    return func.optimal_trajectories_to_return_dictionary(optimal_trajectories)


if __name__ == '__main__':
    folder_name = 'testsets'
    for filename in os.listdir(folder_name):
        fullname = os.path.join(folder_name,filename)
        trajectories, collisions = JSON_IO.read_trajectory_from_json_v2(fullname)
        func.check_for_collisions(cp_sat_solver(trajectories, collisions)['trajectories'])