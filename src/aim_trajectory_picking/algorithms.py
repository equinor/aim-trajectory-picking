import networkx as nx
from ortools.sat.python import cp_model
from ortools.linear_solver import pywraplp 
import matplotlib.pyplot as plt
import numpy as np
from aim_trajectory_picking.util import *
class Trajectory:
    '''
    A class to represent a Trajectory.

    ...

    Attributes:
    -----------
    id: int
        unique id of this trajectory
    donor: str
        the donor (origin) of this particular trajectory
    target: str
        the target (endpoint) if this particular trajectory
    value: int/double
        the value of this particular trajectory in accordance to some cost function
    collisions: List<int>
        A list of trajectory id's with which this trajectory collides. 
        Does not account for trajectories having the same donor/target

    Methods:
    --------
    add_collision(self, trajectory):
        Adds the trajectory to this objects collision list, and adds itself to the other trajectory objects collision list.
    
    add_collision_by_id(self, id):
        Adds the given id to this trajectory's collision list. Does not add itself to the given trajectory id's collision list.


    '''
    def __init__(self,_id, _donor, _target, _value):
        ''' 
        Constructs a trajectory object with an empty collision list.

        Parameters:
        -----------
        id: int
            unique id of this trajectory
        donor: str
            the donor (origin) of this particular trajectory
        target: str
            the target (endpoint) if this particular trajectory
        value: int/double
            the value of this particular trajectory in accordance to some cost function
        '''
        self.id = _id
        self.donor = _donor
        self.target = _target
        self.value = _value
        self.collisions = set()
    
    def add_collision(self, trajectory):
        '''
        Add a collision to this trajectory object. Also adds self to the other trajectory's collision list.

        Parameters:
        -----------
        trajectory: Trajectory
            the trajectory to be added to this objects collision list.
        
        Returns:
        --------
        None
        '''
        if trajectory.id not in self.collisions:
            self.collisions.add(trajectory.id)
            trajectory.add_collision(self)

    def add_collision_by_id(self, id):
        '''
        Add a collision to this trajectory object

        Parameters:
        -----------
        id: int
            the trajectory id to be added to this trajectory objects collission list
        
        Returns:
        --------
        None
        '''
        self.collisions.add(id)
 
    def __str__(self):
        return str(self.id) + ": "+ str(self.donor) + "-->" + str(self.target) + "  Value " + str(self.value) + " "
    
    def __eq__(self, other):
        if not isinstance(other, Trajectory):
            return NotImplemented
        return self.id == other.id and self.donor == other.donor and self.target == other.target and self.value == other.value and self.collisions == other.collisions

    def __hash__(self):
        return self.id 
def general_trajectory_algorithm(graph, choice_function, *, visualize=False):
    '''
    Solved the trajectory picking problem in a 'pseudogreedy' way: some choice_function is passed \
        and this function used that to pick the next node. It then removes the chosen node and \
            all collisions with this one, then picks another. This process repeats until the graph is \
                empty. Optionally plots each step of the process.
    
    Parameters:
    -----------
    graph: nx.Graph()
        A graph where every node is a trajectory and every edge a mutual exclusivity
    choice_function: function(nodes)
        function that takes a list of nodes and returns the 'most optimal' node given certain criteria
    visualize: bool, optional
        if True, each step of the algorithm will be plotted (default is False)

    Returns:
    --------
    dictionary: dict
        a dictionary with the keys 'value' and 'trajectories'. 'value' gives the total value of the trajectories as int, \
            and 'trajectories' gives a list of the 'optimal' trajectory objects found.
    
    '''
    optimal_trajectories = []
    if visualize:
        plt.figure()
    while graph.number_of_nodes() != 0: 
        nodes = list(graph.nodes)
        chosen_node = choice_function(graph) 
        _node_colors = []
        if visualize:
            for i in range(len(nodes)): #color nodes for visual purposes
                if nodes[i] == chosen_node: 
                    _node_colors.append('green')
                elif nodes[i] in graph.neighbors(chosen_node):
                    _node_colors.append('red')
                else:
                    _node_colors.append('blue')
            nx.draw(graph, with_labels=True, node_color=_node_colors)
            plt.show()
        optimal_trajectories.append(chosen_node)
        [graph.remove_node(n) for n in list(graph.neighbors(chosen_node))]
        try:
            graph.remove_node(chosen_node)
        except:
            pass
    dictionary = {}
    dictionary['value'] = sum(n.value for n in optimal_trajectories)
    dictionary['trajectories'] = optimal_trajectories
    return dictionary

def greedy_algorithm(trajectories, collisions, *, visualize=False):
    '''
    Wrapper function for greedy algorithm, utilizing general_trajectory_algorithm internally

    Parameters:
    -----------
    trajectories: List<Trajectory>
        list of trajectories to run greedy algorithm on
    collisions: List<(Trajectory, Trajectory)>:
        list of trajectory collisions
    visualize: bool, optional
        if True the steps of the algorithm will be plotted, if False they will not
    
    Returns:
    dictionary: dict
        a dictionary with the keys 'value' and 'trajectories'. 'value' gives the total value of the trajectories as int, \
            and 'trajectories' gives a list of the 'optimal' trajectory objects found.
    '''
    return general_trajectory_algorithm(create_graph(trajectories, collisions),greedy, visualize=visualize)

def NN_algorithm(trajectories, collisions, *, visualize=False):
    '''
    Wrapper function for number-of-neighbours, utilizing general_trajectory_algorithm internally

    Parameters:
    -----------
    trajectories: List<Trajectory>
        list of trajectories to run number-of-neighbours algorithm on
    collisions: List<(Trajectory, Trajectory)>:
        list of trajectory collisions
    visualize: bool, optional
        if True the steps of the algorithm will be plotted, if False they will not
    
    Returns:
    dictionary: dict
        a dictionary with the keys 'value' and 'trajectories'. 'value' gives the total value of the trajectories as int, \
            and 'trajectories' gives a list of the 'optimal' trajectory objects found.
    '''
    return general_trajectory_algorithm(create_graph(trajectories, collisions),NN_transformation, visualize=visualize)

def weight_transformation_algorithm(trajectories, collisions):
    '''
    Wrapper function for weight-transformation algorithm, utilizing general_trajectory_algorithm internally

    Parameters:
    -----------
    trajectories: List<Trajectory>
        list of trajectories to run weight-transformation algorithm on
    collisions: List<(Trajectory, Trajectory)>:
        list of trajectory collisions
        
    Returns:
    dictionary: dict
        a dictionary with the keys 'value' and 'trajectories'. 'value' gives the total value of the trajectories as int, \
            and 'trajectories' gives a list of the 'optimal' trajectory objects found.
    '''
    return general_trajectory_algorithm(create_graph(trajectories, collisions), weight_transformation)

def random_algorithm(trajectories,collisions, *, visualize=False):
    '''
    Wrapper function for the random algorithm, utilizing general_trajectory_algorithm internally

    Parameters:
    -----------
    trajectories: List<Trajectory>
        list of trajectories to run random algorithm on
    collisions: List<(Trajectory, Trajectory)>:
        list of trajectory collisions
    visualize: bool, optional
        if True the steps of the algorithm will be plotted, if False they will not
    
    Returns:
    dictionary: dict
        a dictionary with the keys 'value' and 'trajectories'. 'value' gives the total value of the trajectories as int, \
            and 'trajectories' gives a list of the 'optimal' trajectory objects found.
            '''
    return general_trajectory_algorithm(create_graph(trajectories, collisions), random_choice, visualize=visualize)

#remove collisions with greedy algo, then do bipartite matching
def bipartite_matching_removed_collisions(trajectories, collisions):
    '''
    This function uses the greedy algorithm to remove any colliding trajectories (not counting target or donor collision),\
        then uses a bipartite max weight matching function to calculate the optimal trajectories.
    
    Parameters:
    -----------
    trajectories: List<Trajectory>
        list of trajectories to constitute the trajectory picking problem
    collisions: List<(Trajectory, Trajectory)>:
        list of trajectory collisions
    Returns:
    --------
    dictionary: dict
        a dictionary with the keys 'value' and 'trajectories'. 'value' gives the total value of the trajectories as int, \
            and 'trajectories' gives a list of the 'optimal' trajectory objects found.
    '''
    G = nx.Graph()
    G.add_nodes_from(trajectories)
    G.add_edges_from(collisions)

    optimal_trajectories_for_matching = general_trajectory_algorithm(G, weight_transformation)
    donors, targets = get_donors_and_targets_from_trajectories(trajectories)
    bi_graph = bipartite_graph(donors, targets, optimal_trajectories_for_matching['trajectories'])
    matching = nx.max_weight_matching(bi_graph)
    
    optimal_trajectories =  get_trajectory_objects_from_matching(matching, trajectories)
    value = sum([t.value for t in optimal_trajectories])
    dictionary = {}
    dictionary['value'] = value
    dictionary['trajectories'] = optimal_trajectories
    return dictionary

def lonely_target_algorithm (trajectories, collisions):
    '''
    Algorithm to solve the trajectory picking problem, focusing on choosing targets only hit by one trajectory.

    Parameters:
    -----------
    trajectories: List<Trajectory>
        list of trajectories to constitute the trajectory picking problem
    collisions: List<(Trajectory, Trajectory)>:
        list of trajectory collisions
    Returns:
    --------
    dictionary: dict
        a dictionary with the keys 'value' and 'trajectories'. 'value' gives the total value of the trajectories as int, \
            and 'trajectories' gives a list of the 'optimal' trajectory objects found.

    '''
    optimal_trajectories = []
    graph = create_graph(trajectories, collisions)
    while graph.number_of_nodes() != 0: 
        lonely_target_trajectories = get_lonely_target_trajectories(list(graph.nodes))
        if len(lonely_target_trajectories) != 0: 
            feasible_nodes = lonely_target_trajectories 
        else: 
            feasible_nodes = list(graph.nodes)
        chosen_node = max(feasible_nodes, key= lambda n : n.value)
        optimal_trajectories.append(chosen_node)
        for n in list(graph.neighbors(chosen_node)): 
            graph.remove_node(n)
        try:
            graph.remove_node(chosen_node)
        except:
            pass
    dictionary = {}
    dictionary['value'] = sum(n.value for n in optimal_trajectories)
    dictionary['trajectories'] = optimal_trajectories
    return dictionary
            

def reversed_greedy(trajectories,collisions, collision_rate = 0.05, last_collisions = bipartite_matching_removed_collisions):
    '''
    Algorithm which follows the inverse logic of the greedy algorithm, focusing on the number of collisions. 
    At each iteration, the trajectory with the highest number of collisions is removed. 
    
    Parameters:
    -----------
    trajectories: List<Trajectory>
        list of trajectories to constitute the trajectory picking problem
    collision_rate: int 
        defaults to 0.05
    collisions: List<(Trajectory, Trajectory)>:
        list of trajectory collisions
    
    Returns:
    --------
    dictionary: dict
        a dictionary with the keys 'value' and 'trajectories'. 'value' gives the total value of the trajectories as int, \
            and 'trajectories' gives a list of the 'optimal' trajectory objects found, after running the result of the
            reverse greedy through the  weight transform algorithm.

    '''
    graph = create_graph(trajectories, collisions)
    highest_collision_trajectory = None
    while highest_collision_trajectory == None or len(highest_collision_trajectory.collisions) > (len(trajectories) * collision_rate):
        if graph.number_of_nodes() == 0:
            break
        for trajectory in list(graph.nodes): 
            num_collisions = len(list(graph.neighbors(trajectory)))
            if highest_collision_trajectory == None or num_collisions > len(highest_collision_trajectory.collisions):
                highest_collision_trajectory = trajectory
        graph.remove_node(highest_collision_trajectory)
    return last_collisions(list(graph.nodes), collisions)

def reversed_greedy_bipartite_matching(trajectories,collisions):
    return reversed_greedy(trajectories,collisions, collision_rate = 0.05, last_collisions = bipartite_matching_removed_collisions)

def reversed_greedy_regular_greedy(trajectories, collisions):
    return reversed_greedy(trajectories,collisions, collision_rate = 0.05, last_collisions = greedy_algorithm)

def reversed_greedy_weight_transformation(trajectories, collisions):
    return reversed_greedy(trajectories,collisions, collision_rate = 0.05, last_collisions = weight_transformation_algorithm)
    
def inverted_minimum_weighted_vertex_cover_algorithm(trajectory,collisons, *, visualize=False):
    '''
    An approximation of a minimum weighted vertex cover performed

    Parameters:
    -----------
    trajectories: list<Trajectory>
        list of trajectories to pick optimal set from

    Returns:
    -----------
    dictionary: dict
        a dictionary with the keys 'value' and 'trajectories'. 'value' gives the total value of the trajectories as int, \
            and 'trajectories' gives a list of the 'optimal' trajectory objects found, after running the result of the
            inverted minimum weighted vertex cover algorithm.
    
    '''
    G = make_graph_with_dictionary_attributes(trajectory)
    trajectory_set = set(trajectory)
    independent_set_nodes = set()
    
    while G.number_of_nodes() > 0:
        vertex_cover_nodes = aprox.min_weighted_vertex_cover(G,weight='value')
        if visualize == True:
            vertex_color = []
            for element in trajectory_set:
                if element not in vertex_cover_nodes:
                    vertex_color.append('blue')
                else:
                    vertex_color.append('red')
            plt.figure()
            nx.draw(G,node_color=vertex_color)
            plt.show()
        current_iteration_independent_set_nodes = trajectory_set.difference(vertex_cover_nodes)
        neighbor_nodes = set()
        for element in current_iteration_independent_set_nodes:
            neighbor_nodes.update(G.neighbors(element))
        G.remove_nodes_from(neighbor_nodes)
        independent_set_nodes.update(current_iteration_independent_set_nodes)
        G.remove_nodes_from(current_iteration_independent_set_nodes)
        trajectory_set = trajectory_set.difference(neighbor_nodes,current_iteration_independent_set_nodes)
    dictionary = translate_trajectory_objects_to_dictionaries(independent_set_nodes)
    return dictionary



def modified_greedy(trajectories,collisions):
    '''
    A version of the greedy algorithm that hopefully has less runtime.

    Parameters:
    -----------
    trajectories: list<Trajectory>
        list of trajectories to pick optimal set from
    collisions: list<Tuple(Trajectory, Trajectory)>
        list of collisions between trajectories

    Returns:
    -----------
    dictionary:
        'value': value of trajectories \n
        'trajectories': list of trajectory objects
    
    '''
    graph = create_graph(trajectories, collisions)
    optimal_trajectories = []
    nodes = list(graph.nodes)
    nodes.sort(key = lambda n: n.value )
    while graph.number_of_nodes() != 0:
        nodes = list(graph.nodes)
        nodes.sort(key = lambda n: n.value)
        chosen_node = nodes[-1]
        #chosen_node = max(node_set, key=lambda t : t.value)
        optimal_trajectories.append(chosen_node)
        for n in list(graph.neighbors(chosen_node)): #remove chosen node and neighbours, given that they are mutually exclusive
            graph.remove_node(n)
            nodes.remove(n)
        #print("finished removing node")
        try:
            graph.remove_node(chosen_node)
        except:
            pass
        try:
            nodes.remove(chosen_node)
        except:
            pass
        #print("added trajectory number: " + str(len(optimal_trajectories)))
    #print("Algorithm: " + choice_function.__name__ + ' sum: ' +str(sum(n.value for n in optimal_trajectories))) #print sum of trajectories
    dictionary = {}
    dictionary['value'] = sum(n.value for n in optimal_trajectories)
    dictionary['trajectories'] = optimal_trajectories
    return dictionary


def invert_and_clique(trajectories, collisions):
    '''
    This function uses clique algorithm to solve the maximal independent weighted set problem, after inverting the graph. Very expensive\
        computationally! Probably infeasible for problem sizes above 200 trajectories.

    From Networkx: The recursive algorithm max_weight_clique may run into recursion depth issues if G contains a clique whose number of nodes is close to the recursion depth limit
    
    Parameters:
    -----------
    trajectories: List<Trajectory>
        list of trajectories to constitute the trajectory picking problem

    Returns:
    --------
    dictionary: dict
        a dictionary with the keys 'value' and 'trajectories'. 'value' gives the total value of the trajectories as int, \
            and 'trajectories' gives a list of the 'optimal' trajectory objects found.
    '''

    G = make_graph_with_dictionary_attributes(trajectories)
    values = nx.get_node_attributes(G,'value')
    G = nx.complement(G)
    nx.set_node_attributes(G,values,'value')
    optimal_trajectories, value = nx.max_weight_clique(G, "value")
    value = sum([t.value for t in optimal_trajectories])

    dictionary = {}
    dictionary['value'] = value
    dictionary['trajectories'] = optimal_trajectories

    return dictionary

def bipartite_matching_not_removed_collisions(trajectories, collisons):
    '''
    This function first uses bipartite matching to find a list of trajectories not colliding in donors or targets.\
        It then removes trajectories colliding in space, and adds new trajectories.
    
    Parameters:
    -----------
    trajectories: List<Trajectory>
        list of trajectories to constitute the trajectory picking problem
    
    Returns:
    --------
    dictionary: dict
        a dictionary with the keys 'value' and 'trajectories'. 'value' gives the total value of the trajectories as int, \
            and 'trajectories' gives a list of the 'optimal' trajectory objects found.
    '''
    donors, targets = get_donors_and_targets_from_trajectories(trajectories)
    bi_graph = bipartite_graph(donors, targets, trajectories)
    matching = nx.max_weight_matching(bi_graph)
    
    optimal_trajectories =  get_trajectory_objects_from_matching(matching, trajectories)
    [trajectories.remove(trajectory) for trajectory in optimal_trajectories]
    
    donors_already_picked = set()
    targets_already_picked = set()
    ids_already_picked = set()
    collisions = []
    
    optimal_trajectories.sort(key = lambda n: n.value )
    for trajectory in optimal_trajectories:
        donors_already_picked.add(trajectory.donor)
        targets_already_picked.add(trajectory.target)
        ids_already_picked.add(trajectory.id)
        collisions.append(trajectory.collisions)
    for trajectory in optimal_trajectories:
        for collision_list in collisions:
            if trajectory.id in collision_list:
                optimal_trajectories.remove(trajectory)
                collisions.remove(collision_list)
    trajectories.sort(key = lambda n: n.value, reverse = True)
    for trajectory in trajectories:
        skip = False
        if (trajectory.donor in donors_already_picked) or (trajectory.target in targets_already_picked):
            continue
        for collision in collision_list:
            if trajectory.id in collision:
                skip = True
                break
        if skip == True:
            continue
        optimal_trajectories.append(trajectory)    
    value = sum([t.value for t in optimal_trajectories])
    dictionary = {}
    dictionary['value'] = value
    dictionary['trajectories'] = optimal_trajectories
    return dictionary

def cp_sat_solver(trajectories, collisions):
    '''
    Function to calculate optimal trajectories for the trajectory picking problem, using the google ortools library internally.
    Method used is integer linear programming, which restates the problem in terms of an objective function, variables, and constraints.
    Parameters:
    -----------
    trajectories: List<Trajectory>
        list of trajectories to constitute the trajectory picking problem
    
    Returns:
    --------
    dictionary: dict
        a dictionary with the keys 'value' and 'trajectories'. 'value' gives the total value of the trajectories as int, \
            and 'trajectories' gives a list of the 'optimal' trajectory objects found.
    '''
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
    #print(sum(t.value for t in optimal_trajectories))
    return optimal_trajectories_to_return_dictionary(optimal_trajectories)




def ILP(trajectories, collisions):
    '''
    Function to calculate optimal trajectories for the trajectory picking problem, using the google ortools library internally.
    Method used is linear optimization, which restates the problem in terms of an objective function, variables, and constraints.
    
    Parameters:
    -----------
    trajectories: List<Trajectory>
        list of trajectories to constitute the trajectory picking problem
    
    Returns:
    --------
    dictionary: dict
        a dictionary with the keys 'value' and 'trajectories'. 'value' gives the total value of the trajectories as int, \
            and 'trajectories' gives a list of the 'optimal' trajectory objects found.
    '''
    
    # retval, _time = func.timer(ILP_formatter,trajectories)
    # print('time to format data: ', _time)
    data, total_value, trajectories = ILP_formatter(trajectories)
    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver('SCIP')

    infinity = solver.infinity()
    x = {}
    for j in range(data['num_vars']):
        x[j] = solver.IntVar(0, 1, 'x[%i]' % j)
    #print('Number of variables =', solver.NumVariables())


    for i in range(data['num_constraints']):
        constraint_expr = \
        [data['constraint_coeffs'][i][j] * x[j] for j in range(data['num_vars'])]
        solver.Add(sum(constraint_expr) >= data['bounds'][i])
    
    #print('Number of constraints =', solver.NumConstraints())

    # Set objective function coefficients
    objective = solver.Objective()
    for j in range(data['num_vars']):
        objective.SetCoefficient(x[j], data['obj_coeffs'][j])
    objective.SetMaximization()

    status = solver.Solve()
    
    optimal_trajectories = []
    for i in range(len(x)):
        if x[i].solution_value() == 1:
            optimal_trajectories.append(trajectories[i])
    return optimal_trajectories_to_return_dictionary(optimal_trajectories)
