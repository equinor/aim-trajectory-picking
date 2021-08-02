import random
import numpy as np
from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt
import time
from itertools import permutations

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
        trajectories.append(Trajectory(i, random.choice(donors), random.choice(targets),random.randint(0,data_range))) 
    collisions = []
    for i in range(int(num_trajectories*collision_rate)):
        collisions.append((trajectories[np.random.randint(0,num_trajectories)],trajectories[np.random.randint(0,num_trajectories)]))
    for pair in collisions:
        pair[0].add_collision(pair[1])
    return donors, targets, trajectories, collisions

def create_realistic_data(num_donors, num_targets, num_trajectories, collision_rate=0,data_range=100 ):
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
    targets =[]
    trajectories = []
    for i in range(num_trajectories):
        donor = random.randint(0, num_donors)
        donors.append(donor)
        target = random .randint(max(0, donor - num_targets//5), min(donor + num_targets//5, num_targets-1))
        targets.append(target)
        trajectories.append(Trajectory(i, donor, target,random.randint(0,data_range))) 
    for i in range(num_trajectories):
        for j in range(i, num_trajectories):
            if i !=j and trajectories[i].donor != trajectories[j].donor and trajectories[i].target != trajectories[j].target:
                if trajectories[i].donor in list(range(trajectories[j].donor - num_donors//10, trajectories[j].donor + num_donors//5)):
                    collision_rate = 0.05
                elif trajectories[i].donor in list(range(trajectories[j].donor - num_donors//6, trajectories[j].donor + num_donors//5)):
                    collision_rate = 0.02
                elif trajectories[i].donor in list(range(trajectories[j].donor - num_donors//4, trajectories[j].donor + num_donors//3)):
                    collision_rate = 0.01           
                if np.random.binomial(1,collision_rate):
                    trajectories[i].add_collision(trajectories[j])
    return donors, targets, trajectories

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
        node_set.append(Trajectory(element.id, element.donor, element.target,element.value))
    dictionary = {}
    dictionary['value'] = sum(n.value for n in node_set)
    dictionary['trajectories'] = node_set
    return dictionary

def optimal_trajectories_to_return_dictionary(optimal_trajectories):

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