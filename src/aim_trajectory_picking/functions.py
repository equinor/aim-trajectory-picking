import networkx as nx
import random
from networkx.algorithms.link_analysis.pagerank_alg import google_matrix
import numpy as np
import matplotlib.pyplot as plt
import time
from networkx.algorithms import approximation as aprox
from itertools import combinations

# A class to define all the useful information pertaining a certain trajectory.
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
    
    #Add a collision to the trajectory object. Also adds self to the other trajectory's collision list.
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

    #Add a collision by id only, does not add itself to the other trajectory's collision list.
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
        return str(self.id) + ": "+ self.donor + "-->" + self.target + "  Value " + str(self.value) + " "
    
    def __eq__(self, other):
        if not isinstance(other, Trajectory):
            # don't attempt to compare against unrelated types
            return NotImplemented
        return self.id == other.id and self.donor == other.donor and self.target == other.target and self.value == other.value and self.collisions == other.collisions

    def __hash__(self):
        return self.id #+ self.value

#Data is created in the following way: Amount is given and the function returns the nodes and trajectories between these.
#Also returns collisions with given collision rate
def create_data(no_donors, no_targets, no_trajectories, collision_rate=0,data_range=100 ):
    '''
    Creates a dataset of the correct format for the trajectory picking problem.

    The dataset consists of donors, targets, and trajectories

    Parameters:
    ----------
        no_donors: int
            a positive integer corresponding to the number of donors desired
        no_targets: int
            a positive integer corresponding to the number of targets desired
        no_trajectories: int
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
    for i in range(no_donors):
        donors.append("D" + str(i)) # name all donors
    for i in range(no_targets):
        targets.append("T"+str(i)) # name all targets
    for i in range(no_trajectories):
        #a trajectory is remembered by a tuple of start and end, a value, and id and collisions
        trajectories.append(Trajectory(i, random.choice(donors), random.choice(targets),random.randint(0,data_range))) 
    #loop through all pairs of trajectories, and randomly choose that they collide
    for i in range(no_trajectories):
        for j in range(i, no_trajectories):
            #if the trajectories are different and dont already collide based on target and donor:
            if i !=j and trajectories[i].donor != trajectories[j].donor and trajectories[i].target != trajectories[j].target:
                if np.random.binomial(1,collision_rate):
                    trajectories[i].add_collision(trajectories[j])
    return donors, targets, trajectories

#creates and returns a colored bypartite graph, with the option of showing it
def bipartite_graph(donors, targets, trajectories, visual=False):
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
    visual: bool, optional
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
    for i in range(len(donors)):
        _node_color.append('green')
    for i in range(len(targets)):
        _node_color.append('red')
    if visual:
        plt.figure()
        pos = nx.bipartite_layout(g, donors)
        nx.draw(g, pos, node_color=_node_color, with_labels=True)
        labels = nx.get_edge_attributes(g,'weight')
        nx.draw_networkx_edge_labels(g,pos,edge_labels=labels)
        plt.show()
    return g
    
#Returns true if trajectories collide/are mutually exclusive, false otherwise
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


#Creates a graph where the nodes are trajectories and the edges are mutual exclusivity, either through collisions in space or in donors/targets
#Returns the created graph

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


def create_graph(trajectories, collisions):
    '''
    Algorithm to create graph from trajectories with the given collisions. Created to try and reduce graph creation times.

    Parameters:
    -----------
    trajectories: list<Trajectory>
        list of trajectories to create graph from
    collisions: list<Tuple(Trajectory, Trajectory)>
        list of tuples which contain trajectory objects that collide

    Returns:
    --------
    G: nx.Graph()
        graph with trajectories as nodes and mutual exclusivity as edges
    '''
    G = nx.Graph()
    G.add_nodes_from(trajectories)
    print("done adding nodes")
    donor_and_target_collisions = {}
    for t in trajectories:
        try:
            donor_and_target_collisions[t.donor].append(t)
        except:
            donor_and_target_collisions[t.donor] = []
            donor_and_target_collisions[t.donor].append(t)
        try:
            donor_and_target_collisions[t.target].append(t)
        except:
            donor_and_target_collisions[t.target] = []
            donor_and_target_collisions[t.target].append(t)
    for key in donor_and_target_collisions:
        donor_or_target_collisions = donor_and_target_collisions[key]
        for pair in list(combinations(donor_or_target_collisions, 2)):
            G.add_edge(pair[0], pair[1])
    for pair in collisions:
        G.add_edge(pair[0], pair[1])
    return G
        
def make_transformed_graph_from_trajectory_dictionaries(trajectories):
    '''
    Helper function to create graph with node attributes for usage of built in nx functions.

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


#Computes a pseudogreedy optimal set of trajectories, given a function to determine the next node to add.
def abstract_trajectory_algorithm(graph, choice_function,visualize=False):
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
    while graph.number_of_nodes() != 0: #while there still are nodes left
        nodes = list(graph.nodes)
        chosen_node = choice_function(graph) #choose most optimal node based on given choice function
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
        # for n in list(graph.neighbors(chosen_node)): #remove chosen node and neighbours, given that they are mutually exclusive
        #     graph.remove_node(n)
        [graph.remove_node(n) for n in list(graph.neighbors(chosen_node))]
        graph.remove_node(chosen_node)
        #print("added trajectory number: " + str(len(optimal_trajectories)))
    #print("Algorithm: " + choice_function.__name__ + ' sum: ' +str(sum(n.value for n in optimal_trajectories))) #print sum of trajectories
    dictionary = {}
    dictionary['value'] = sum(n.value for n in optimal_trajectories)
    dictionary['trajectories'] = optimal_trajectories
    return dictionary

#Choice function for pseudogreedy algorithm. Finds the best node by scaling all values by dividing with the sum of blocked nodes.
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
            #print(type(nodes[i]))
            value_adjacent_nodes += n.value
        transformed_weights.append(nodes[i].value /value_adjacent_nodes)
    
    return nodes[transformed_weights.index(max(transformed_weights))]

#Greedy algorithm choice function
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

#Random choice function, used for benchmarks
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

#Scale weights by dividing every node with the amount of nodes it blocks.
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

#Helper function to check for collisions given an optimal trajectory list. Used for determining if the algorithms work correctly.
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

def timer(func, *args, **kwargs):
    ''' Time a function with the given args and kwargs'''
    start = time.perf_counter()
    func(*args, **kwargs)
    stop = time.perf_counter()
    return stop-start

def greedy_algorithm(trajectories, visualize=False):
    '''
    Wrapper function for greedy algorithm, utilizing abstract_trajectory_algorithm internally

    Parameters:
    -----------
    trajectories: List<Trajectory>
        list of trajectories to run greedy algorithm on
    visualize: bool, optional
        if True the steps of the algorithm will be plotted, if False they will not
    
    Returns:
    dictionary: dict
        a dictionary with the keys 'value' and 'trajectories'. 'value' gives the total value of the trajectories as int, \
            and 'trajectories' gives a list of the 'optimal' trajectory objects found.
    '''
    return abstract_trajectory_algorithm(transform_graph(trajectories),greedy, visualize)

def NN_algorithm(trajectories, visualize=False):
    '''
    Wrapper function for number-of-neighbours, utilizing abstract_trajectory_algorithm internally

    Parameters:
    -----------
    trajectories: List<Trajectory>
        list of trajectories to run number-of-neighbours algorithm on
    visualize: bool, optional
        if True the steps of the algorithm will be plotted, if False they will not
    
    Returns:
    dictionary: dict
        a dictionary with the keys 'value' and 'trajectories'. 'value' gives the total value of the trajectories as int, \
            and 'trajectories' gives a list of the 'optimal' trajectory objects found.
    '''
    return abstract_trajectory_algorithm(transform_graph(trajectories),NN_transformation, visualize)

def weight_transformation_algorithm(trajectories, visualize=False):
    '''
    Wrapper function for weight-transformation algorithm, utilizing abstract_trajectory_algorithm internally

    Parameters:
    -----------
    trajectories: List<Trajectory>
        list of trajectories to run weight-transformation algorithm on
    visualize: bool, optional
        if True the steps of the algorithm will be plotted, if False they will not
    
    Returns:
    dictionary: dict
        a dictionary with the keys 'value' and 'trajectories'. 'value' gives the total value of the trajectories as int, \
            and 'trajectories' gives a list of the 'optimal' trajectory objects found.
    '''
    return abstract_trajectory_algorithm(transform_graph(trajectories), weight_transformation)

def random_algorithm(trajectories, visualize=False):
    '''
    Wrapper function for the random algorithm, utilizing abstract_trajectory_algorithm internally

    Parameters:
    -----------
    trajectories: List<Trajectory>
        list of trajectories to run random algorithm on
    visualize: bool, optional
        if True the steps of the algorithm will be plotted, if False they will not
    
    Returns:
    dictionary: dict
        a dictionary with the keys 'value' and 'trajectories'. 'value' gives the total value of the trajectories as int, \
            and 'trajectories' gives a list of the 'optimal' trajectory objects found.
            '''
    return abstract_trajectory_algorithm(transform_graph(trajectories), random_choice, visualize)

#remove collisions with greedy algo, then do bipartite matching
def bipartite_matching_removed_collisions(trajectories, visualize):
    '''
    This function uses the greedy algorithm to remove any colliding trajectories (not counting target or donor collision),\
        then uses a bipartite max weight matching function to calculate the optimal trajectories.
    
    Parameters:
    -----------
    trajectories: List<Trajectory>
        list of trajectories to constitute the trajectory picking problem
    visualize: bool, optional
        if True, plot every step of algorithm
    
    Returns:
    --------
    dictionary: dict
        a dictionary with the keys 'value' and 'trajectories'. 'value' gives the total value of the trajectories as int, \
            and 'trajectories' gives a list of the 'optimal' trajectory objects found.
    '''
    G = nx.Graph()
    #print(type(trajectories))
    G.add_nodes_from(trajectories)
    # Add collisions from donors and targets
    for i in range(len(trajectories)):
        for j in range(i, len(trajectories)):
            if i != j:
                if trajectories[i].id in  trajectories[j].collisions:
                    G.add_edge(trajectories[i], trajectories[j])
    
    optimal_trajectories_for_matching = abstract_trajectory_algorithm(G, greedy, False)
    donors, targets = get_donors_and_targets_from_trajectories(trajectories)
    bi_graph = bipartite_graph(donors, targets, optimal_trajectories_for_matching['trajectories'])
    matching = nx.max_weight_matching(bi_graph)
    
    optimal_trajectories =  get_trajectory_objects_from_matching(matching, trajectories)
    value = sum([t.value for t in optimal_trajectories])
   # for t in optimal_trajectories:
  #      print(t)
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
        

def lonely_target_algorithm (trajectories, visualize=False):
    '''
    Algorithm to solve the trajectory picking problem, focusing on choosing targets only hit by one trajectory.

    Parameters:
    -----------
    trajectories: List<Trajectory>
        list of trajectories to constitute the trajectory picking problem
    visualize: bool, optional
        if True, plot every step of algorithm
    
    Returns:
    --------
    dictionary: dict
        a dictionary with the keys 'value' and 'trajectories'. 'value' gives the total value of the trajectories as int, \
            and 'trajectories' gives a list of the 'optimal' trajectory objects found.

    '''
    optimal_trajectories = []
    graph = transform_graph(trajectories)
    while graph.number_of_nodes() != 0: 
        lonely_target_trajectories = get_lonely_target_trajectories(list(graph.nodes))
        if len(lonely_target_trajectories) != 0: 
            feasible_nodes = lonely_target_trajectories 
        else: 
            feasible_nodes = list(graph.nodes)
        chosen_node = max(feasible_nodes, key= lambda n : n.value)
        optimal_trajectories.append(chosen_node)
        for n in list(graph.neighbors(chosen_node)): #remove chosen node and neighbours, given that they are mutually exclusive
            graph.remove_node(n)
        graph.remove_node(chosen_node)
    dictionary = {}
    dictionary['value'] = sum(n.value for n in optimal_trajectories)
    dictionary['trajectories'] = optimal_trajectories
    return dictionary
            

def reversed_greedy(trajectories, visualize=False, collision_rate = 0.05, last_collisions = bipartite_matching_removed_collisions):
    '''
    Algorithm which follows the inverse logic of the greedy algorithm, focusing on the number of collisions. 
    At each iteration, the trajectory with the highest number of collisions is removed. 
    
    Parameters:
    -----------
    trajectories: List<Trajectory>
        list of trajectories to constitute the trajectory picking problem
    visualize: bool, optional
        if True, plot every step of algorithm
    collision_rate: int 
        defaults to 0.05
    
    Returns:
    --------
    dictionary: dict
        a dictionary with the keys 'value' and 'trajectories'. 'value' gives the total value of the trajectories as int, \
            and 'trajectories' gives a list of the 'optimal' trajectory objects found, after running the result of the
            reverse greedy through the  weight transform algorithm.

    '''
    graph = transform_graph(trajectories)
    highest_collision_trajectory = None
    while highest_collision_trajectory == None or len(highest_collision_trajectory.collisions) > (len(trajectories) * collision_rate):
        if graph.number_of_nodes() == 0:
            break
        for tra in list(graph.nodes): 
            num_collisions = len(list(graph.neighbors(tra)))
            if highest_collision_trajectory == None or num_collisions > len(highest_collision_trajectory.collisions):
                highest_collision_trajectory = tra
        graph.remove_node(highest_collision_trajectory)
    #return greedy_algorithm(list(graph.nodes))
    #return weight_transformation_algorithm(list(graph.nodes))
    return last_collisions(list(graph.nodes), False)

def reversed_greedy_bipartite_matching(trajectories, visualize=False):
    return reversed_greedy(trajectories, visualize, collision_rate = 0.05, last_collisions = bipartite_matching_removed_collisions)

def reversed_greedy_regular_greedy(trajectories, visualize=False):
    return reversed_greedy(trajectories, visualize, collision_rate = 0.05, last_collisions = greedy_algorithm)

def reversed_greedy_weight_transformation(trajectories, visualize=False):
    return reversed_greedy(trajectories, visualize, collision_rate = 0.05, last_collisions = weight_transformation_algorithm)


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

def inverted_minimum_weighted_vertex_cover_algorithm(trajectory, visualize=False):
    '''
    An approximation of a minimum weighted vertex cover performed on a inverted graph

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
    G = make_transformed_graph_from_trajectory_dictionaries(trajectory)
    G = invert_graph(G)
    vertex_cover_nodes = aprox.min_weighted_vertex_cover(G,weight='value')
    dictionary = translate_trajectory_objects_to_dictionaries(vertex_cover_nodes)
    return dictionary



def modified_greedy(trajectories,collisions, visualize=False):
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
    print("started making graph")
    start = time.perf_counter()
    graph = create_graph(trajectories, collisions)
    stop = time.perf_counter()
    print("done creating graph with time: " + str(start-stop))
    optimal_trajectories = []
    nodes = list(graph.nodes)
    print("started sorting")
    nodes.sort(key = lambda n: n.value )
    print("finished sorting")
    while graph.number_of_nodes() != 0:
        chosen_node = nodes[-1]
        optimal_trajectories.append(chosen_node)
        print("started removing node")
        for n in list(graph.neighbors(chosen_node)): #remove chosen node and neighbours, given that they are mutually exclusive
            graph.remove_node(n)
            nodes.remove(n)
        print("finished removing node")
        graph.remove_node(chosen_node)
        print("added trajectory number: " + str(len(optimal_trajectories)))
    #print("Algorithm: " + choice_function.__name__ + ' sum: ' +str(sum(n.value for n in optimal_trajectories))) #print sum of trajectories
    dictionary = {}
    dictionary['value'] = sum(n.value for n in optimal_trajectories)
    dictionary['trajectories'] = optimal_trajectories


def maximum_independent_set_algorithm(trajectory,visualize=False):
    G = transform_graph(trajectory)
    max_ind_set = aprox.maximum_independent_set(G)
    return max_ind_set

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


def invert_and_clique(trajectories, visualize = False):
    '''
    This function uses clique algorithm to solve the maximal independent weighted set problem, after inverting the graph. Very expensive\
        computationally! Probably infeasible for problem sizes above 200 trajectories.
    
    Parameters:
    -----------
    trajectories: List<Trajectory>
        list of trajectories to constitute the trajectory picking problem
    visualize: bool, optional
        if True, plot every step of algorithm
    
    Returns:
    --------
    dictionary: dict
        a dictionary with the keys 'value' and 'trajectories'. 'value' gives the total value of the trajectories as int, \
            and 'trajectories' gives a list of the 'optimal' trajectory objects found.
    '''

    G = make_transformed_graph_from_trajectory_dictionaries(trajectories)
    G = invert_graph(G)
    optimal_trajectories, value = nx.max_weight_clique(G, "value")
    value = sum([t.value for t in optimal_trajectories])

    dictionary = {}
    dictionary['value'] = value
    dictionary['trajectories'] = optimal_trajectories

    return dictionary

'''
# Pseudocode for the reversed greedy where we first do bipartite matching and then solve for collisions 
def bipartite_first_collisions_seconds(trajectories, visualize=False)
          
    use bipartite_matching to solve for donors and targets
    while there still are collisions in the picked trajectories: 
        remove the colliding trajectories of lowest value and replace them with other trajectories with same donor and target,
        so that there still are no collisions in donors or targets.
        if the added trajectories, that replaced former colliding trajectories, don't collide with other picked trajectories:
            break:
    return trajectories
'''

def bipartite_matching_not_removed_collisions(trajectories, visualize):
    '''
    This function uses bipartite matching to find a list of trajectories not colliding in donors or targets.\
        It then removes trajectories colliding in space, and adds new trajectories.
    
    Parameters:
    -----------
    trajectories: List<Trajectory>
        list of trajectories to constitute the trajectory picking problem
    visualize: bool, optional
        if True, plot every step of algorithm
    
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
    [trajectories.remove(tra) for tra in optimal_trajectories]
    
   #Hjelp til fjern trajectories som kolliderer med hverandre fra optimal_trajectories med reversed greedy
   
   #legger til brønnbaner med donor og target som ikke allerede er plukket ut
    donors_already_picked = set()
    targets_already_picked = set()
    ids_already_picked = set()
    collisions = []
    
    #collision_dict = dict()
    optimal_trajectories.sort(key = lambda n: n.value )
    for tra in optimal_trajectories:
        #for id in tra.collisions:
            # if id in collision_dict.keys:
            #     collision_dict[str(id)] +=1
            # else:
            #     collision_dict[str(id)] = 0
        donors_already_picked.add(tra.donor)
        targets_already_picked.add(tra.target)
        ids_already_picked.add(tra.id)
        collisions.append(tra.collisions)
    for tra in optimal_trajectories:
        for collision_list in collisions:
            if tra.id in collision_list:
                optimal_trajectories.remove(tra)
                collisions.remove(collision_list)
    #Er trajectories en liste eller et dictionary. Om dictionary må nøklene sorteres mhp. value
    trajectories.sort(key = lambda n: n.value, reverse = True)
    for tra in trajectories:
        skip = False
        if (tra.donor in donors_already_picked) or (tra.target in targets_already_picked):
            continue
        for collision in collision_list:
            if tra.id in collision:
                skip = True
                break
        if skip == True:
            continue
        optimal_trajectories.append(tra)    
    value = sum([t.value for t in optimal_trajectories])
    dictionary = {}
    dictionary['value'] = value
    print(value)
    dictionary['trajectories'] = optimal_trajectories
    return dictionary