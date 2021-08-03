import random
from networkx import algorithms
from networkx.algorithms.chordal import _max_cardinality_node
import numpy as np
from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt
import time
from itertools import permutations
import os
import pandas as pd

from aim_trajectory_picking import JSON_IO

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
    def __init__(self,_id, _true_id, _donor, _target, _value):
        ''' 
        Constructs a trajectory object with an empty collision list.

        Parameters:
        -----------
        id: int
            id for this trajectory used for list indexing
        true_id: int
            unique id of this trajectory
        donor: str
            the donor (origin) of this particular trajectory
        target: str
            the target (endpoint) if this particular trajectory
        value: int/double
            the value of this particular trajectory in accordance to some cost function
        '''
        self.id = _id
        self.true_id = _true_id
        self.donor = _donor
        self.target = _target
        self.value = _value
        self.collisions = []
    
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
            self.collisions.append(trajectory.id)
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
        self.collisions.append(id)
 
    def __str__(self):
        return str(self.id) + ": "+ str(self.donor) + "-->" + str(self.target) + "  Value " + str(self.value) + " collisions :" + str(self.collisions)
    
    def __eq__(self, other):
        if not isinstance(other, Trajectory):
            return NotImplemented
        return self.id == other.id and self.donor == other.donor and self.target == other.target and self.value == other.value and self.collisions == other.collisions

    def __hash__(self):
        return self.id 

def create_graph(trajectories, collisions):
    '''
    Creates a graph by adding nodes and edges. 

    Parameters:
    -----------
    trajectories: List<Trajectory>
        list of trajectory objects to be added to the graph
    collisions: List<Tuple(Trajectory, Trajectory)>
        list of tuples with colliding trajectory objects
        
    Returns:
    --------
    G: nx.Graph()
        graph with trajectories as nodes.\
             All nodes have a dictionary of their attributes attached
    
    '''
    G = nx.Graph()
    G.add_nodes_from(trajectories)
    G.add_edges_from(collisions)
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
    for donor in donor_dict:
        G.add_edges_from([item for item in permutations(donor_dict[donor],2) ])
    for target in target_dict:
        G.add_edges_from([item for item in permutations(target_dict[target],2) ])
    return G


def create_data(num_donors, num_targets, num_trajectories, collision_rate=0.05, data_range=100 ):
    '''
    Creates a dataset of the correct format for the trajectory picking problem.

    The dataset consists of donors, targets, and trajectories

    Parameters:
    ----------
        num_donors: int
            a positive integer corresponding to the number of donors desired
        num_targets: int
            a positive integer corresponding to the number of targets desired
        num_trajectories: int
            a positive integer corresponding to the number of trajectories desired
        collision_rate: float, optional
            a positive floating point number corresponding to the percentage probability that two trajectories collide (default is 0)
        data_range: int, optional
            a positive integer corresponding to the maximal value of the Trajectory.value field (default is 100)

    Returns:
    --------
        donors: List<str>
            a list of the names of the donors in the dataset
        targets: List<str>
            a list of the names of the targets in the dataset
        trajectories: List<Trajectory>
            a list of the Trajectory objects created
    '''
    donors = []
    targets = []
    trajectories = []
    for i in range(num_donors):
        donors.append("D" + str(i))
    for i in range(num_targets):
        targets.append("T"+str(i)) 
    for i in range(num_trajectories):
        trajectories.append(Trajectory(i,i, random.choice(donors), random.choice(targets),random.randint(0,data_range))) 
    collisions = []
    for i in range(int(num_trajectories*collision_rate)):
        collisions.append((trajectories[np.random.randint(0,num_trajectories)],trajectories[np.random.randint(0,num_trajectories)]))
    for pair in collisions:
        pair[0].add_collision(pair[1])
    return donors, targets, trajectories, collisions


def create_realistic_data(num_donors, num_targets, num_trajectories, collision_rate=0.05,data_range=100 ):
    '''
    Creates a dataset of the correct format for the trajectory picking problem.

    The dataset consists of donors, targets, and trajectories, and a list of collisions.
    The dataset is in multiple ways attempted to be more realistic than the dataset created from create_data(),
    as explained in the document file in GitHub:

    Parameters:
    ----------
        num_donors: int
            a positive integer corresponding to the number of donors desired
        num_targets: int
            a positive integer corresponding to the number of targets desired
        num_trajectories: int
            a positive integer corresponding to the number of trajectories desired
        collision_rate: float, optional
            a positive floating point number corresponding to the percentage probability that two trajectories collide (default is 0.05)
        data_range: int, optional
            a positive integer corresponding to the maximal value of the Trajectory.value field (default is 100)

    Returns:
    --------
        donors: List<str>
            a list of the names of the donors in the dataset
        targets: List<str>
            a list of the names of the targets in the dataset
        trajectories: List<Trajectory>
            a list of the Trajectory objects created
    '''
    donors = []
    targets =[]
    trajectories = []
    # Notice that the donors and targets are represented by integers instead of strings. This is
    # to make it easier to generate targets based on donors
    for i in range(num_trajectories):
        donor = random.randint(0, num_donors)
        donors.append(donor)
        # Target is related to the donor. The respective donors and targets of a trajectory is to be imgained as fairly close to each
        # other. This is because the selectable trajectories in a realistic dataset typically will be short,
        # since the cost of them depends on their length. I this function, the position of donors and
        # targets are implied by their ids, which is sort of imagined to represent it's location.
        # As an example, if there are 50 targets and 50 donors, then trajectories of donor id qual to 15
        # will typically go to a target id close to 15. However, if there are 50 targets and 200 trajectories,
        # then a trajectory of donor id 15 will typically go to a target id close to 15*(200/50)
        target = random.randint(max(0, donor*(int(num_targets/num_donors)) - int(num_targets*0.1)), min(donor*(int(num_targets//num_donors)) + int(num_targets*0.1, num_targets-1)))
        targets.append(target)
        trajectories.append(Trajectory(i, donor, target,random.randint(0,data_range))) 
    in_collision = 0
    collisions = []
    for trajectory in trajectories:
        collision_each_trajectory = []
        in_collision = random.randint(0, num_trajectories)
        # If a trajectory collides with other trajectories, then a list of trajectories close to it is
        # made by appending trajectories with donors and targets close to the relevant trajectory. Then
        # the trajectory it collides with is randomly picked from this list, and the collision is appended
        # to a list of collisions.
        if in_collision < num_trajectories*collision_rate:
            for trajectory2 in trajectories:
                if trajectory.donor != trajectory2.donor and trajectory.target != trajectory2.target:
                    if (trajectory2.donor > (trajectory.donor - num_donors*0.10) and trajectory2.donor < (trajectory.donor + num_donors*0.10)) or (trajectory2.target > (trajectory.target - num_targets*0.10) and trajectory2.target < (trajectory.target + num_targets*0.10)):
                        collision_each_trajectory.append(trajectory2)
        # Adds three collisions if the trajectory collides with three other trajectories and has more than
        # or equal to three trajectories close to itself
        if in_collision < num_trajectories*(collision_rate**3) and len(collision_each_trajectory) >= 3:
            colliding_trajectory1 = random.choice(collision_each_trajectory)
            collision_each_trajectory.remove(colliding_trajectory1)
            colliding_trajectory2 = random.choice(collision_each_trajectory)
            collision_each_trajectory.remove(colliding_trajectory2)
            colliding_trajectory3 = random.choice(collision_each_trajectory)
            collisions.append((trajectory, colliding_trajectory1))
            collisions.append((trajectory, colliding_trajectory2))
            collisions.append((trajectory, colliding_trajectory3))
        # Adds two collisions if the trajectory collides with three other trajectories and has more than
        # or equal to two trajectories close to itself
        elif in_collision < num_trajectories*(collision_rate**2) and len(collision_each_trajectory) >= 2:
            colliding_trajectory1 = random.choice(collision_each_trajectory)
            collision_each_trajectory.remove(colliding_trajectory1)
            colliding_trajectory2 = random.choice(collision_each_trajectory)
            collisions.append((trajectory, colliding_trajectory1))
            collisions.append((trajectory, colliding_trajectory2))
        # Adds one collision if the trajectory collides with another trajectory and has more than
        # or equal to one trajectory close to itself
        elif in_collision < num_trajectories*collision_rate and len(collision_each_trajectory) >= 1:
            colliding_trajectory1 = random.choice(collision_each_trajectory)
            collisions.append((trajectory, colliding_trajectory1))    
    for pair in collisions:
        pair[0].add_collision(pair[1])
    return donors, targets, trajectories, collisions

def bipartite_graph(donors, targets, trajectories, *, visualize=False):
    '''
    Creates and returns a bipartite graph from the trajectories, donors and targets. Optionally plots the graph.

    This function doesn't need to take donors and targets as arguments, but if they are given as arguments \
        the function doesnt need to spend time finding them from Trajectory objects.
    
    Parameters:
    -----------
    donors: List<str>
        list of donor names, one partition of the graph
    targets: List<str>
        list of target names, the other partition of the graph
    trajectories: List<Trajectory>
        list of Trajectory objects to be transformed into bipartite graph
    visualize: bool, optional
        the function plots the graph if this boolean is true.
    
    Returns:
    --------
        g: nx.Graph()
            a bipartite graph with the donors + targets as nodes and trajectories as edges
    '''
    g = nx.Graph()
    g.add_nodes_from(donors + targets)
    for t in trajectories:
        g.add_edge(t.donor, t.target, weight=t.value)
    _node_color =[]
    if visualize:
        for i in range(len(donors)):
            _node_color.append('green')
        for i in range(len(targets)):
            _node_color.append('red')
        plt.figure()
        pos = nx.bipartite_layout(g, donors)
        nx.draw(g, pos, node_color=_node_color, with_labels=True)
        labels = nx.get_edge_attributes(g,'weight')
        nx.draw_networkx_edge_labels(g,pos,edge_labels=labels)
        plt.show()
    return g

def mutually_exclusive_trajectories(t1, t2):
    '''
    A simple function to detect if two trajectories exclude eachother by colliding in some way. \
        Tests donor collision, target collision, and path collision

    Parameters:
    -----------
    t1: Trajectory
        a trajectory to test mutual exclusivity
    t2: Trajectory
        a trajectory to test mutual exclusivity

    Returns:
    --------
    bool: 
        True if the trajectories collide, False if they dont
    '''
    if(t1.donor == t2.donor):
        return True
    if(t1.target == t2.target):
        return True
    if(t2.id in t1.collisions):
        return True
    return False


def transform_graph(trajectories):
    '''
    Creates a graph from the given trajectories such that every trajectory is a node and every collision/mutual exclusivity is an edge.

    Parameters:
    -----------
    trajectories: List<Trajectory>
        list of trajectories to create graph from

    Returns:
    --------
    G: nx.Graph()
        graph where every node is a trajectory and every edge is a collision
    '''
    G = nx.Graph()
    G.add_nodes_from(trajectories)
    for i in range(len(trajectories)):
        for j in range(i, len(trajectories)):
            if i != j:
                if mutually_exclusive_trajectories(trajectories[i], trajectories[j]):
                    G.add_edge(trajectories[i], trajectories[j])
    return G

        
def make_graph_with_dictionary_attributes(trajectories):
    '''
    Helper function to create graph with node attributes for usage of built-in nx functions.

    Parameters:
    -----------
    trajectories: list<Trajectory>
        list of trajectories to create graph from

    Returns:
    --------
    G: nx.Graph()
        graph with trajectories as nodes and mutual exclusivity as edges. All nodes have a dictionary of their attributes attached
    '''
    G = nx.Graph()
    G.add_nodes_from(trajectories)
    node_attrs = {}
    for i in range(len(trajectories)):
        node_attrs[trajectories[i]] = trajectories[i].__dict__
    nx.set_node_attributes(G,node_attrs)
    for i in range(len(trajectories)):
        for j in range(i, len(trajectories)):
            if i != j:
                if mutually_exclusive_trajectories(trajectories[i], trajectories[j]):
                    G.add_edge(trajectories[i], trajectories[j])
    return G

def timer(func, *args, **kwargs):
    ''' Time a function with the given args and kwargs'''
    start = time.perf_counter()
    retval = func(*args, **kwargs)
    stop = time.perf_counter()
    difference = stop-start
    #print(retval)
    return retval, difference

def get_donors_and_targets_from_trajectories(trajectories):
    '''
    Helper function to extract the donors and targets from a list of trajectories.

    Parameters:
    -----------
    trajectories: List<Trajectory>
        list of trajectories from which to extract donors and targets

    Returns:
    --------
    donors: List<str>
        list of names of donors
    targets: List<str>
        list of names of targets
    
    '''
    donors = []
    targets = []
    for t in trajectories:
        if t.donor not in donors:
            donors.append(t.donor)
        if t.target not in targets:
            targets.append(t.target)
    return donors, targets

def get_trajectory_objects_from_matching(matching, trajectories):
    '''
    Helper function to translate results from nx.max_weight_matching(graph) to a list of trajectory objects.

    The matching function only returns node tuples, therefore this function has to match the node tuples to the correct\
        trajectory objects.
    
    Parameters:
    -----------
    trajectories: List<Trajectory>
        list fo trajectory objects that was used to calculate the matching
    matching: set((str, str))
        set of str tuples, constitutes a max_weight_matching of a graph
    
    Returns:
    trajectories_optimal: List<Trajectory>
        list of trajectory objects that make up the max weight matching
    '''
    trajectories_optimal = []
    for t in trajectories:
        if (t.donor, t.target) in matching or (t.target , t.donor) in matching:
            add_trajectory = True
            for i in trajectories_optimal:
                if i.donor == t.donor and i.target == t.target:
                    add_trajectory = False
                    if i.value < t.value:
                        trajectories_optimal.remove(i) 
                        trajectories_optimal.append(t)
                        break   
                    else:
                        break
            if add_trajectory:
                trajectories_optimal.append(t)
    return trajectories_optimal

def get_lonely_target_trajectories (trajectories):
    '''
    Function to get the trajectories of targets only hit once.

    Parameters:
    -----------
    trajectories: List<Trajectory>
        list of trajectories from which the lonely target hitter will be found
    
    Returns:
    --------
    tra_list: List<Trajectory>
        list of trajectories which are the only ones to hit their respective targets
    '''
    target_dict = {}
    tra_list = []
    for tra in trajectories: 
        if tra.target in target_dict:
            target_dict[tra.target].append(tra)
        else: 
            target_dict[tra.target] = [tra]
    for target in target_dict: 
        if len(target_dict[target]) == 1: 
            tra_list.append(target_dict[target][0])
    return tra_list

def translate_trajectory_objects_to_dictionaries(trajectories):
    '''
    Parameters:
    -----------
    trajectories: set<Trajectory>
        set of trajectories

    Returns:
    -----------
    dictionary: dict
        a dictionary with the keys 'value' and 'trajectories'. 'value' gives the total value of the trajectories as int, \
            and 'trajectories' gives a list of the 'optimal' trajectory objects.
    
    '''
    node_set = []
    for element in trajectories:
        node_set.append(Trajectory(element.id,element.id, element.donor, element.target,element.value))
    dictionary = {}
    dictionary['value'] = sum(n.value for n in node_set)
    dictionary['trajectories'] = node_set
    return dictionary

def optimal_trajectories_to_return_dictionary(optimal_trajectories):
    '''
    Making a dictionary which stores optimal trajectories. 
    
    Parameters:
    -----------
    optimal_trajectories: List<Trajectory>
        list of trajectories

    Returns:
    -----------
    dictionary: dict
        a dictionary with the keys 'value' and 'trajectories'. 'value' gives the total value of the trajectories as int, \
            and 'trajectories' gives a list of the 'optimal' trajectory objects.

    '''

    value = sum([t.value for t in optimal_trajectories])
    dictionary = {}
    dictionary['value'] = value
    dictionary['trajectories'] = optimal_trajectories
    return dictionary

def invert_graph(graph):
    '''
    Function to invert a graph in order to use Clique algorithm to solve maximal independent weighted set problem.
    Probably O(n^2)

    Parameters:
    -----------
    graph: nx.Graph()
        graph to be inverted (invert edges)

    Returns:
    --------
    graph: nx.Graph()
        the inverted graph

    '''
    for pair in combinations(graph.nodes, 2): 
        if graph.has_edge(pair[0], pair[1]): 
            graph.remove_edge(pair[0], pair[1])
        else: 
            graph.add_edge(pair[0], pair[1]) 
    return graph 

def check_for_collisions(optimal_trajectories):
    '''
    Helper function to ensure no trajectories collide in the final answer.

    Parameters:
    -----------
    optimal_trajectores: List<Trajectory>
        list of trajectories to check if feasible ( no collisions)
    
    Returns:
    --------
    bool:
        True if there is an internal collision, False if not
    '''
    donors =[]
    targets =[]
    ids = []
    for t in optimal_trajectories:
        if t.donor in donors:
            print(donors)
            print(t.donor)
            print("error in donors")
            return True
        if t.target in targets:
            print("error in targets")
            return True
        elif t in ids:
            print("error in collisions")
            return True
        donors.append(t.donor)
        targets.append(t.target)
        ids += t.collisions
    return False


def weight_transformation(graph):
    '''
    Finds the most 'optimal' node by dividing each node with the value of the nodes it blocks, then picks the node \
        with the most value.
    
    Parameters:
    -----------
    nodes: List<Trajectory>
        list of Trajectory objects of which the most 'optimal' will be calculated
    
    Returns: 
    --------
    node: 
        the most optimal Trajectory
    '''
    nodes = list(graph.nodes)
    transformed_weights = []
    for i in range(len(nodes)):
        value_adjacent_nodes = 1
        for n in graph.neighbors(nodes[i]):
            value_adjacent_nodes += n.value
        transformed_weights.append(nodes[i].value /value_adjacent_nodes)
    
    return nodes[transformed_weights.index(max(transformed_weights))]

def greedy(graph):
    '''
    Finds the most 'optimal' picking the one with the highest value.
    
    Parameters:
    -----------
    nodes: List<Trajectory>
        list of Trajectory objects of which the most 'optimal' will be calculated
    
    Returns: 
    --------
    node: 
        the most optimal Trajectory
    '''
    nodes = list(graph.nodes)
    return  max(nodes, key= lambda n : n.value)

def random_choice(graph):
    '''
    Returns a random node. Used for benchmarks

    Parameters:
    -----------
    nodes: List<Trajectory>
        list of Trajectory objects of which the most 'optimal' will be calculated
    
    Returns: 
    --------
    node: 
        the most optimal Trajectory
    '''
    nodes = list(graph.nodes)
    return random.choice(nodes)

def NN_transformation(graph):
    '''
    Finds the most 'optimal' by scaling every value by the amount of nodes adjacent, then picking the node with \
        the highest value.
    
    Parameters:
    -----------
    nodes: List<Trajectory>
        list of Trajectory objects of which the most 'optimal' will be calculated
    
    Returns: 
    --------
    node: 
        the most optimal Trajectory
    '''
    nodes = list(graph.nodes)
    transformed_weights = []
    for i in range(len(nodes)):
        number_of_adjacent_nodes = 1
        for n in graph.neighbors(nodes[i]):
            number_of_adjacent_nodes += 1
        transformed_weights.append(nodes[i].value /number_of_adjacent_nodes)
    
    return nodes[transformed_weights.index(max(transformed_weights))]


def get_donor_and_target_collisions(trajectories):
    '''
    A function which extracts colliding donors and targets from a list of trajectories. 

    Parameters:
    ----------
    trajectories: list<Trajectory>

    Returns: 
    -------
    donor_dict: dict
        dictionary with colliding donors
    target_dict: dict
        dictionary with colliding targets
    '''
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

    
def ILP_formatter(trajectories):
    '''
    Function to format data correctly for the ILP solver. 

    Parameters:
    -----------
    trajectories: List<Trajectory>
    list of trajectory objects of which the optimal will be found

    Returns:
    -------
    None
    '''
    data = {}
    donor_dict , target_dict = get_donor_and_target_collisions(trajectories)
    
    
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

def save_optimal_trajectories_to_file(results, filename,data_names):
    '''
    Function to sift through a results dictionary and save the optimal trajectory set to a file.

    Parameters:
    -----------
    results: dictionary[algorithm.__name__][dataset] = {
                'value': int
                'trajectories: list<Trajectory>
                'runtime':float
    filename: str
    Filename of file to store result in

    Returns:
    -------
    optimal_trajectories: dictionary{str : list<Int>}
    Dictionary in which the dataset name are the keys and the corresponding most optimal trajectories are the values.
    '''
    max_value = 0
    optimal_trajectories = {}
    max_value_dataset = {}
    for algorithm_name in results:
        for dataset_name in results[algorithm_name]: #data_names
            if dataset_name not in max_value_dataset.keys():
                max_value_dataset[dataset_name] = 0
            if dataset_name in results[algorithm_name].keys() and results[algorithm_name][dataset_name]['value'] > max_value_dataset[dataset_name]:
                max_value_dataset[dataset_name] =results[algorithm_name][dataset_name]['value']
                if  isinstance(results[algorithm_name][dataset_name]['trajectories'][0], dict):
                     optimal_trajectories[dataset_name] = [e['true_id'] for e in results[algorithm_name][dataset_name]['trajectories']]
                else:
                    optimal_trajectories[dataset_name] = [e.true_id for e in results[algorithm_name][dataset_name]['trajectories']]
    for dataset_name in max_value_dataset:
        if max_value_dataset[dataset_name] == 0:
            print("Optimal trajectories not found for dataset: ", dataset_name)
    JSON_IO.write_data_to_json_file(filename, optimal_trajectories)
    return optimal_trajectories

def get_datasets(dataset_folders, algorithms,refresh, filename='results.txt'):
    '''
    Function to find and/or create the given data and return it as a list.

    Parameters:
    dataset_folders: list<str>
    list of folders if which the datasets will be read and added to the list.

    algorithms: list<Function>
    list of algorithms which will be run on the given datasets. They exist to reduce runtime by not reading datasets which already have saved results

    refresh: bool
    bool to indicate if previously saved data will be ignored (if True) and results recalculated

    filename: str
    filename of the file to be read from

    Returns:
    --------
    data: list< Tuple( list<Trajectory>, list< Tuple(Trajectory, Trajectory)>)
    list of trajectories and their collisions

    dataset_names: list<str>
    list of dataset_names, either read (when reading from file) or None (when random data is chosen)
    '''
    no_datasets = False
    data = []
    dataset_names = []
    if dataset_folders == None:
        print("None-type input file, bringing up runtime benchmarks")
        dataset_folders = []
        dataset_folders.append('testsets')
    try:
        if dataset_folders[0] == 'random':
            print("random data generation chosen")
            data = []
            num_donors = int(dataset_folders[1])
            num_targets = int(dataset_folders[2])
            num_trajectories = int(dataset_folders[3])
            collision_rate = float(dataset_folders[4])
            if len(dataset_folders) < 5:
                num_datasets = 1
            else:
                num_datasets = int(dataset_folders[5])
            for i in range(num_datasets):
                print("making dataset nr: " + str(i))
                _,_,trajectories, collisions = create_data(num_donors, num_targets, num_trajectories, collision_rate)
                data.append((trajectories, collisions))
                dataset_names.append('dataset_' + str(i)+ '.txt')
            return data, None
        elif dataset_folders[0] == 'increasing':
            upper_limit_trajectories = 10000
            print("increasing data generation chosen")
            data = []
            num_donors = int(dataset_folders[1])
            num_targets = int(dataset_folders[2])
            initial_num_trajectories = int(dataset_folders[3])
            collision_rate = float(dataset_folders[4])
            if len(dataset_folders) < 5:
                num_datasets = 1
            else:
                num_datasets = int(dataset_folders[5])
            for i in range(num_datasets):
                if initial_num_trajectories * (i+1) > upper_limit_trajectories:
                    break
                print("making dataset nr: " + str(i))
                _,_,trajectories, collisions = create_data(num_donors, num_targets, initial_num_trajectories * (i + 1), collision_rate)
                data.append((trajectories, collisions))
                dataset_names.append('dataset_' + str(i)+ '.txt')
            return data, None
        else:
            prev_results = get_previous_results(filename)
            datasets_as_string = ' '.join(map(str, dataset_folders))
            if len(os.listdir(datasets_as_string))==0:
                no_datasets = True
            else:
                no_datasets = False
                for folder in dataset_folders:
                    for filename in os.listdir(folder):
                        print("else file")
                        if refresh or not all(algo.__name__ in prev_results.keys() for algo in algorithms) or not all(filename in prev_results[algo.__name__].keys() for algo in algorithms):
                            fullpath = os.path.join(folder,filename)
                            data.append(JSON_IO.read_trajectory_from_json(fullpath))
                            dataset_names.append(filename)
                            print(fullpath)
                        else:
                            dataset_names.append(filename)

    except Exception as e:
        print("exception thrown:", str(e))
        if len(data) == 0:
            print("Dataset arguments not recognized, reading from testsets instead.")
            for filename in os.listdir('testsets'):
                fullpath = os.path.join('testsets',filename)
                data.append(JSON_IO.read_trajectory_from_json(fullpath))
                dataset_names.append(filename)
    return data, dataset_names, no_datasets

def addlabels(x,y):
    '''
    Adding text to bars in bar charts. 
    '''
    for i in range(len(x)):
        plt.text(i,y[i],y[i])

def plot_results_with_runtimes(algorithms, results, _dataset_names=0, *,show_figure=True):
    '''
    Fully automatic function that plots the results per algorithm. 

    Important: the names of the algorithms must be keys in the results dictionary, and every value is a list \
        that consists of dictionaries, which again contain the value of trajectories of that specific algorithm on that \
            specific dataset.
    
    Parameters:
    -----------
    algorithms: List<Function>
        list of functions used to obtain results

    results: dictionary[algorithm.__name__][dataset] = {
                'value': int
                'trajectories: list<Trajectory>
                'runtime':float
    } 

    Returns:
    --------
    None
    '''
    means = []
    if _dataset_names == 0 or _dataset_names == None:
        dataset_names = [str(i) for i in range(len(results[algorithms[0].__name__]))]
    else:
        dataset_names = _dataset_names

    algo_names = [e.__name__ for e in algorithms]
    algo_runtimes = []

    if show_figure:
        print("Not showing plots chosen")
    else:
        if len(dataset_names) > 1:
            fig, axs = plt.subplots(2,1, figsize=(10,5))
            ax1 = axs[0]
            ax2 = axs[1]
            ax1.set_xlabel('Datasets')
            ax2.set_xlabel('Datasets')
            ax1.set_ylabel('Value')
            ax2.set_ylabel('Runtime (seconds)')
            ax1.title.set_text('Algorithm Performance')
            ax2.title.set_text('Algorithm Runtime')
            fig.tight_layout(pad=3)
            for algorithm in algorithms:
                results_per_dataset = [results[algorithm.__name__][dataset_name]['value'] for dataset_name in dataset_names]
                algo_runtimes =  [results[algorithm.__name__][dataset_name]['runtime'] for dataset_name in dataset_names]

                ax1.plot(dataset_names, results_per_dataset, label=algorithm.__name__)
                ax1.scatter(dataset_names, results_per_dataset, s=5, alpha=0.5) 
                ax2.plot(dataset_names, algo_runtimes, '--',label=algorithm.__name__)
                ax2.scatter(dataset_names, algo_runtimes, s=5, alpha=0.5)

                means.append(np.mean(results_per_dataset))
            leg1 = ax1.legend()
            leg1.set_draggable(state=True)
            # plt.xticks(rotation=45)
            leg2 = ax2.legend()
            leg2.set_draggable(state=True)
            plt.show()
        else:
            plt.figure()
            for algorithm in algorithms:
                results_per_dataset = [results[algorithm.__name__][dataset_name]['value'] for dataset_name in dataset_names]
                algo_runtimes =  [results[algorithm.__name__][dataset_name]['runtime'] for dataset_name in dataset_names]
                means.append(np.mean(results_per_dataset))
                plt.scatter(dataset_names, algo_runtimes, s=10, alpha=0.5)
            plt.xlabel('Algorithm Name')
            plt.ylabel('Runtime (seconds)')
            plt.title('Runtime graph')
            leg = plt.legend(algo_names)
            leg.set_draggable(state=True)
            plt.show()
        plt.figure(figsize=(12, 6))
        plt.bar(algo_names, means, color=(0.2, 0.4, 0.6, 0.6))
        addlabels(algo_names, means)
        for i, (name, height) in enumerate(zip(algo_names,  means)):
            plt.text(i, height/2, ' ' + name,
                ha='center', va='center', rotation=-90, fontsize=10)
        plt.xticks([])
        plt.title('Average Algorithm Performance')
        plt.xlabel('Algorithm Name')
        plt.ylabel('Average Value')
        plt.show()

def get_previous_results(filename):
    '''
    Function to read previous results from given file, if found.

    Parameters:
    -----------
    filename: str
    name of file previous results are located in

    Returns:
    prev_results: dictionary[algorithm.__name__][dataset] = {
                'value': int
                'trajectories: list<Trajectory>
                'runtime':float
    } 
    '''
    try:
        prev_results = JSON_IO.read_value_trajectories_runtime_from_file(filename)
    except:
        prev_results = {}
    return prev_results

def calculate_or_read_results(algos, _datasets,refresh, *, _is_random=False, filename='results.txt', _dataset_names=None):
    '''
    Function to either calculate the specified results or read them from file (if they have been calculated before)

    Parameters:
    algos: list<Function>
    list of functions to be ran on _datasets. Must accept list<Trajectory> and list<Tuple(Trajectory, Trajectory)> (collisions) and return\
        a result dictionary.

    _datasets: list< Tuple( list<Trajectory>, list< Tuple(Trajectory, Trajectory)>)
    list of trajectories and their collisions
    
    refresh: bool
    bool to indicate whether to recalculate results even if previously calculated.

    _is_random: bool
    bool to indicate if dataset is randomly generated or not, and therefore does not need to be saved to file

    filename: str
    file to read previous results from

    _dataset_names: list<str>
    list of dataset names, to be used in indexing dictionary. If random data generation is chosen, this will be None.

    Returns:
    --------
    results: dictionary[algorithm.__name__][dataset] = {
                'value': int
                'trajectories: list<Trajectory>
                'runtime':float
    } 
    '''

    dataset_names = [str(i) for i in range(len(_datasets))] if _dataset_names == None else _dataset_names

    prev_results = dict()
    if not _is_random:
        prev_results = get_previous_results(filename)

    for algorithm in algos:
        if algorithm.__name__ not in prev_results.keys():
            prev_results[algorithm.__name__] = {}

    for data in _datasets:
        data_name = dataset_names[_datasets.index(data)]
        for algorithm in algos:
            if not refresh and algorithm.__name__ in prev_results.keys() and _dataset_names!=None and data_name in prev_results[algorithm.__name__].keys():
                print("algorithm " + algorithm.__name__ + " on dataset " + data_name + " already in " + filename)
            else:
                #print(type(data))
                answer, runtime = timer(algorithm, data[0], data[1])
                answer['runtime'] = runtime
                prev_results[algorithm.__name__][data_name] = answer
                print("done with algorithm: " + algorithm.__name__ + " on dataset " + data_name)

    #check that trajectories are feasible
    for name in algos:
        for dataset in prev_results[name.__name__]:
            if check_for_collisions(prev_results[name.__name__][dataset]['trajectories']):
                print("error in algorithm" + name.__name__)

    if _dataset_names != None:
        JSON_IO.write_value_trajectories_runtime_from_file( prev_results, filename)
    return prev_results

def find_best_performing_algorithm(results, algorithms, used_datasets):
    '''
    Function finding the best algorithm 
    
    Returns: 
    -------
    None
    '''
    best_result = 0
    algorithm_finder = 0
    best_algorithm_name_list = []
    matrix_list = []
    all_datasets_list =[]
    listToStr_list = []
    best_algorithm_name = 0

    for algorithm in algorithms:
        for all_datasets in results[algorithm.__name__]:
            if all_datasets not in all_datasets_list:
                all_datasets_list.append(all_datasets)

    used_datasets_set = set(used_datasets)
    all_datasets_set = set(all_datasets_list)
    intersection = used_datasets_set.intersection(all_datasets_set)
    intersection_as_list = sorted(list(intersection))

    for algorithm in algorithms:
        ram_list = []
        for element in intersection_as_list:
            if element in results[algorithm.__name__]:
                chosen_results_per_dataset = [results[algorithm.__name__][element]['value']]
                ram_list.append(chosen_results_per_dataset[0])

                if sum(chosen_results_per_dataset) > best_result:
                    best_result = sum(chosen_results_per_dataset)
                    for key in results.keys():
                        best_algorithm_name_list.append(key)
                    best_algorithm_name = best_algorithm_name_list[algorithm_finder]
        algorithm_finder += 1
        matrix_list.append(ram_list)
    map_matrix = list(map(max, zip(*matrix_list)))
    algorithm_finder_per_dataset = 0
    best_performing_algorithms = [[] for x in range(len(map_matrix))]
    for n in range(len(map_matrix)):
        for m in range(len(matrix_list)):
            if map_matrix[n] == matrix_list[m][n]:
                best_performing_algorithms[n].append(best_algorithm_name_list[m])
    for j in range(len(best_performing_algorithms)):
        listToStr = ' '.join(map(str, best_performing_algorithms[j]))
        listToStr_list.append(listToStr)
    return(intersection_as_list, listToStr_list, map_matrix, best_algorithm_name, best_result)

def translate_results_to_dict(results, algorithms):
    '''
    Translates the results to a dictionary used for plotting.

    Parameters:
    -----------
    results: dict
        file of results with one big dictionary 

    algorithms: dict 
        dictionary with algorithms as elements

    '''
    results_as_dict = {}
    for algo in algorithms:
        name = algo.__name__
        results_as_dict[name] = [d['value'] for d in results[name]]
    return results_as_dict

def plot_algorithm_values_per_dataset(algorithms, results, directory): 
    results_dict = {}
    for algorithm in algorithms: 
        results_dict[algorithm.__name__ ] = 0

    dataset_names = [i for i in range(4)]
    pandas_dict = translate_results_to_dict(results, algorithms)
    plotdata = pd.DataFrame(
        pandas_dict, 
        index=dataset_names
    )

    plotdata.plot(kind="bar", cmap =plt.get_cmap('Pastel1'))
    plt.title("Performance of Algorithms on Datasets")
    plt.xlabel("Dataset")
    plt.ylabel("Value")
    plt.show()