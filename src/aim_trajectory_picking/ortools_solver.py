from ortools.linear_solver import pywraplp
from aim_trajectory_picking import JSON_IO
import os

def format_data_from_file(fullname):
    trajectories = JSON_IO.read_trajectory_from_json(fullname)
    
    data = {}
    
    data['constraint_coeffs'] = []
    for element in trajectories:
        colls = [0] * len(trajectories)
        colls[element.id] = -1
        for collision in element.collisions:
            colls[collision] = -1
        data['constraint_coeffs'].append(colls)
    data['bounds'] = [-1] * len(trajectories)
    obj_coeffs = []
    for element in trajectories:
        obj_coeffs.append(element.value)
    total_value = sum(obj_coeffs)
    data['obj_coeffs'] = obj_coeffs
    data['num_vars'] = len(trajectories)
    data['num_constraints'] = len(trajectories)
    return data, total_value


def integer_linear_program(fullname):

    data, total_value = format_data_from_file(fullname)
    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver('SCIP')

    infinity = solver.infinity()
    x = {}
    for j in range(data['num_vars']):
        x[j] = solver.IntVar(0, infinity, 'x[%i]' % j)
    print('Number of variables =', solver.NumVariables())

    # Set the contraint bounds
    for i in range(data['num_constraints']):
        constraint = solver.RowConstraint(data['bounds'][i], 0, '')
        for j in range(data['num_vars']):
            constraint.SetCoefficient(x[j], data['constraint_coeffs'][i][j])
    print('Number of constraints =', solver.NumConstraints())

    # In Python, you can also set the constraints as follows.
    # for i in range(data['num_constraints']):
    #  constraint_expr = \
    # [data['constraint_coeffs'][i][j] * x[j] for j in range(data['num_vars'])]
    #  solver.Add(sum(constraint_expr) <= data['bounds'][i])

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
        for j in range(data['num_vars']):
            print(x[j].name(), ' = ', x[j].solution_value())
        # print()
        print('\nProblem solved in %f milliseconds' % solver.wall_time())
        print('Problem solved in %d iterations' % solver.iterations())
        print('Problem solved in %d branch-and-bound nodes' % solver.nodes())

        print("\nMaximum weighted independent set value = ", total_value - solver.Objective().Value())
    else:
        print('The problem does not have an optimal solution.')


if __name__ == '__main__':
    folder_name = 'testsets'
    for filename in os.listdir(folder_name):
        fullname = os.path.join(folder_name,filename)
        integer_linear_program(fullname)