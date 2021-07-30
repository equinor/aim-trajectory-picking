import itertools
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import time
from networkx.algorithms import approximation as aprox
from itertools import combinations
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


def create_data(num_donors, num_targets, num_trajectories, collision_rate=0.05,data_range=10000 ):
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


def bipartite_graph(donors, targets, trajectories,collisions, *,visualize=False):
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
        for n in list(graph.neighbors(chosen_node)):
            if n != chosen_node:
                graph.remove_node(n)
        graph.remove_node(chosen_node)
    dictionary = {}
    dictionary['value'] = sum(n.value for n in optimal_trajectories)
    dictionary['trajectories'] = optimal_trajectories
    return dictionary

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
        ids.append(t.collisions)
    return False

def timer(func, *args, **kwargs):
    ''' Time a function with the given args and kwargs'''
    start = time.perf_counter()
    return_value = func(*args, **kwargs)
    stop = time.perf_counter()
    time_used = stop-start
    return return_value, time_used

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
    bi_graph = bipartite_graph(donors, targets, optimal_trajectories_for_matching['trajectories'], collisions)
    matching = nx.max_weight_matching(bi_graph)
    
    optimal_trajectories =  get_trajectory_objects_from_matching(matching, trajectories)
    value = sum([t.value for t in optimal_trajectories])
    dictionary = {}
    dictionary['value'] = value
    dictionary['trajectories'] = optimal_trajectories
    return dictionary

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
    trajectory_list: List<Trajectory>
        list of trajectories which are the only ones to hit their respective targets
    '''
    target_dict = {}
    trajectory_list = []
    for trajectory in trajectories: 
        if trajectory.target in target_dict:
            target_dict[trajectory.target].append(trajectory)
        else: 
            target_dict[trajectory.target] = [trajectory]
    for target in target_dict: 
        if len(target_dict[target]) == 1: 
            trajectory_list.append(target_dict[target][0])
    return trajectory_list
        

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
        graph.remove_node(chosen_node)
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

def translate_trajectory_objects_to_dictionaries(trajectories_set):
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
    node_list = []
    for element in trajectories_set:
        node_list.append(Trajectory(element.id, element.donor, element.target,element.value))
    dictionary = {}
    dictionary['value'] = sum(n.value for n in node_list)
    dictionary['trajectories'] = node_list
    return dictionary

def optimal_trajectories_to_return_dictionary(optimal_trajectories):

    value = sum([t.value for t in optimal_trajectories])
    dictionary = {}
    dictionary['value'] = value
    dictionary['trajectories'] = optimal_trajectories
    return dictionary


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
        graph.remove_node(chosen_node)
        nodes.remove(chosen_node)
        #print("added trajectory number: " + str(len(optimal_trajectories)))
    #print("Algorithm: " + choice_function.__name__ + ' sum: ' +str(sum(n.value for n in optimal_trajectories))) #print sum of trajectories
    dictionary = {}
    dictionary['value'] = sum(n.value for n in optimal_trajectories)
    dictionary['trajectories'] = optimal_trajectories
    return dictionary

# Skal denne slettes??
def maximum_independent_set_algorithm(trajectory):

    G = transform_graph(trajectory)
    max_ind_set = aprox.maximum_independent_set(G)
    trajectory = translate_trajectory_objects_to_dictionaries(max_ind_set)
    return trajectory

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
        G.add_edges_from([item for item in itertools.permutations(donor_dict[donor],2) ])
    for target in target_dict:
        G.add_edges_from([item for item in itertools.permutations(target_dict[target],2) ])
    return G


def create_graph(trajectories, collisions):
    G = nx.Graph()
   # G.add_nodes_from(trajectories)
    donor_dict = {}
    target_dict = {}
    for t in trajectories:
        G.add_node(t)
        if t.donor in donor_dict:
            donor_dict[t.donor].append(t)
        else:
            donor_dict[t.donor] = [t]
        if t.target in target_dict:
            target_dict[t.target].append(t)
        else:
            target_dict[t.target] = [t]
    for donor in donor_dict:
        G.add_edges_from(itertools.combinations(donor_dict[donor],2) )
    for target in target_dict:
        G.add_edges_from(itertools.combinations(target_dict[target],2))
    G.add_edges_from(collisions)
    return G

if __name__ == '__main__':
    traj, col = JSON_IO.read_trajectory_from_json_v2(r'even_datasets\even_test_5.txt')
    # graph, time_ = timer(create_graph, traj, col)
    # print("time to create graph w collisions", time_)
    # _graph, time_ = timer(transform_graph, traj)
    # print("time to create graph without collisions", time_)
    
    result, _time = timer( modified_greedy,traj, col)
    print("modified greedy time:" , _time)
    result, _time = timer(greedy_algorithm, traj,col)
    print('greedy time:',_time)

