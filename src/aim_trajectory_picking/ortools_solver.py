from ortools.linear_solver import pywraplp
from aim_trajectory_picking import JSON_IO
import os
from aim_trajectory_picking import functions as func

def format_data_from_file(fullname):
    trajectories = JSON_IO.read_trajectory_from_json(fullname)
    
    data = {}
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
    
    
    data['constraint_coeffs'] = []
    num_trajectories = len(trajectories)
    for element in trajectories:
        colls = [0] * len(trajectories)
        if len(element.collisions) != 0:
            colls[element.id] = -1
            for collision in element.collisions:
                colls[collision] = -1
            data['constraint_coeffs'].append(colls)

    for target in target_dict:
        if len(target_dict[target]) < 2:
            continue
        target_constraint = [0] * num_trajectories
        for trajectory in target_dict[target]:
            target_constraint[trajectory.id] = -1
        data['constraint_coeffs'].append(target_constraint)
    '''
    dictionary {
        'D1': [2, 4, 6]
    }
    '''
    for donor in donor_dict:
        if len(donor_dict[donor]) < 2:
            continue
        donor_constraint = [0]*num_trajectories
        for trajectory in donor_dict[donor]:
            donor_constraint[trajectory.id] = -1
        #print("donor constraints:", donor_constraint)
        data['constraint_coeffs'].append(donor_constraint)

    data['bounds'] = [-1] * len(data['constraint_coeffs'])
    obj_coeffs = []
    for element in trajectories:
        obj_coeffs.append(element.value)
    total_value = sum(obj_coeffs)
    data['obj_coeffs'] = obj_coeffs
    data['num_vars'] = len(trajectories)
    data['num_constraints'] = len(data['constraint_coeffs'])
    return data, total_value, trajectories


def integer_linear_program(fullname):

    data, total_value, trajectories = format_data_from_file(fullname)
    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver('SCIP')

    infinity = solver.infinity()
    x = {}
    for j in range(data['num_vars']):
        x[j] = solver.IntVar(0, 1, 'x[%i]' % j)
    print('Number of variables =', solver.NumVariables())

    # Set the contraint bounds
    # for i in range(data['num_constraints']):
    #     constraint = solver.RowConstraint(data['bounds'][i], 0, '')
    #     for j in range(data['num_vars']):
    #         constraint.SetCoefficient(x[j], data['constraint_coeffs'][i][j])
    

    # In Python, you can also set the constraints as follows.
    for i in range(data['num_constraints']):
        constraint_expr = \
        [data['constraint_coeffs'][i][j] * x[j] for j in range(data['num_vars'])]
        solver.Add(sum(constraint_expr) >= data['bounds'][i])
    
    print('Number of constraints =', solver.NumConstraints())

    # Set objective function coefficients
    objective = solver.Objective()
    for j in range(data['num_vars']):
        objective.SetCoefficient(x[j], data['obj_coeffs'][j])
    objective.SetMaximization()

    # In Python, you can also set the objective as follows.
    # obj_expr = [data['obj_coeffs'][j] * x[j] for j in range(data['num_vars'])]
    # solver.Maximize(solver.Sum(obj_expr))

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        print('\nObjective value =', solver.Objective().Value())
        # for j in range(data['num_vars']):
        #     print(x[j].name(), ' = ', x[j].solution_value())
        # print()
        print('\nProblem solved in %f milliseconds' % solver.wall_time())
        print('Problem solved in %d iterations' % solver.iterations())
        print('Problem solved in %d branch-and-bound nodes' % solver.nodes())

        print("\nMaximum weighted independent set value = ", solver.Objective().Value())
    else:
        print('The problem does not have an optimal solution.')
    
    optimal_trajectories = []
    for i in range(len(x)):
        if x[i].solution_value() == 1:
            optimal_trajectories.append(trajectories[i])
    #[print(t.donor, t.target) for t in optimal_trajectories]
    #print(sum(t.value for t in optimal_trajectories))
    return func.optimal_trajectories_to_return_dictionary(optimal_trajectories)


def ILP(trajectories, collisions):
    
    # retval, _time = func.timer(ILP_formatter,trajectories)
    # print('time to format data: ', _time)
    data, total_value, trajectories = ILP_formatter(trajectories)
    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver('SCIP')

    infinity = solver.infinity()
    x = {}
    for j in range(data['num_vars']):
        x[j] = solver.IntVar(0, 1, 'x[%i]' % j)
    print('Number of variables =', solver.NumVariables())


    for i in range(data['num_constraints']):
        constraint_expr = \
        [data['constraint_coeffs'][i][j] * x[j] for j in range(data['num_vars'])]
        solver.Add(sum(constraint_expr) >= data['bounds'][i])
    
    print('Number of constraints =', solver.NumConstraints())

    # Set objective function coefficients
    objective = solver.Objective()
    for j in range(data['num_vars']):
        objective.SetCoefficient(x[j], data['obj_coeffs'][j])
    objective.SetMaximization()

    status = solver.Solve()

    # if status == pywraplp.Solver.OPTIMAL:
    #     print('\nObjective value =', solver.Objective().Value())

    #     print('\nProblem solved in %f milliseconds' % solver.wall_time())
    #     print('Problem solved in %d iterations' % solver.iterations())
    #     print('Problem solved in %d branch-and-bound nodes' % solver.nodes())

    #     print("\nMaximum weighted independent set value = ", solver.Objective().Value())
    # else:
    #     print('The problem does not have an optimal solution.')
    
    optimal_trajectories = []
    for i in range(len(x)):
        if x[i].solution_value() == 1:
            optimal_trajectories.append(trajectories[i])
    return func.optimal_trajectories_to_return_dictionary(optimal_trajectories)

def ILP_formatter(trajectories):
    data = {}
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
    
    
    data['constraint_coeffs'] = []
    num_trajectories = len(trajectories)
    for element in trajectories:
        colls = [0] * len(trajectories)
        if len(element.collisions) != 0:
            colls[element.id] = -1
            for collision in element.collisions:
                colls[collision] = -1
            data['constraint_coeffs'].append(colls)

    for target in target_dict:
        if len(target_dict[target]) < 2:
            continue
        target_constraint = [0] * num_trajectories
        for trajectory in target_dict[target]:
            target_constraint[trajectory.id] = -1
        data['constraint_coeffs'].append(target_constraint)
    '''
    dictionary {
        'D1': [2, 4, 6]
    }
    '''
    for donor in donor_dict:
        if len(donor_dict[donor]) < 2:
            continue
        donor_constraint = [0]*num_trajectories
        for trajectory in donor_dict[donor]:
            donor_constraint[trajectory.id] = -1
        #print("donor constraints:", donor_constraint)
        data['constraint_coeffs'].append(donor_constraint)

    data['bounds'] = [-1] * len(data['constraint_coeffs'])
    obj_coeffs = []
    for element in trajectories:
        obj_coeffs.append(element.value)
    total_value = sum(obj_coeffs)
    data['obj_coeffs'] = obj_coeffs
    data['num_vars'] = len(trajectories)
    data['num_constraints'] = len(data['constraint_coeffs'])
    return data, total_value, trajectories

if __name__ == '__main__':
    folder_name = 'datasets2'
    for filename in os.listdir(folder_name):
        fullname = os.path.join(folder_name,filename)
        func.check_for_collisions(integer_linear_program(fullname)['trajectories'])