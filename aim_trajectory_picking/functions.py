import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt

# A class to define all the useful information pertaining a certain trajectory.
class Trajectory:
    def __init__(self,_id, _donor, _target, _value):
        self.id = _id
        self.donor = _donor
        self.target = _target
        self.value = _value
        self.collisions = []
    
    #Add a collision to the trajectory object. Also adds self to the other trajectory's collision list.
    def add_collision(self, trajectory):
        if trajectory.id not in self.collisions:
            self.collisions.append(trajectory.id)
            trajectory.add_collision(self)

    #Add a collision by id only, does not add itself to the other trajectory's collision list.
    def add_collision_by_id(self, id):
        self.collisions.append(id)
 
    def __str__(self):
        return str(self.id) + ": "+ self.donor + "-->" + self.target + "  Value " + str(self.value) + " "
    
    def __eq__(self, other):
        if not isinstance(other, Trajectory):
            # don't attempt to compare against unrelated types
            return NotImplemented
        return self.id == other.id and self.donor == other.donor and self.target == other.target and self.value == other.value and self.collisions == other.collisions

    def __hash__(self):
        return self.id + self.value

#Data is created in the following way: Amount is given and the function returns the nodes and trajectories between these.
#Also returns collisions with given collision rate
def create_data(no_donors, no_targets, no_trajectories, collision_rate=0 ):
    donors = []
    targets = []
    trajectories = []
    for i in range(no_donors):
        donors.append("D" + str(i)) # name all donors
    for i in range(no_targets):
        targets.append("T"+str(i)) # name all targets
    for i in range(no_trajectories):
        #a trajectory is remembered by a tuple of start and end, a value, and id and collisions
        trajectories.append(Trajectory(i, random.choice(donors), random.choice(targets),random.randint(0,10))) 
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
    g = nx.Graph()
    g.add_nodes_from(donors+ targets)
    for t in trajectories:
        g.add_edge(t.donor, t.target, weight=t.value)
    _node_color =[]
    for i in range(len(donors)):
        _node_color.append('green')
    for i in range(len(targets)):
        _node_color.append('red')
    if visual:
        plt.figure()
        nx.draw(g, nx.bipartite_layout(g,donors), node_color=_node_color, with_labels=True)
        plt.show()
    return g
    
#Returns true if trajectories collide/are mutually exclusive, false otherwise
def mutually_exclusive_trajectories(t1, t2):
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
    G = nx.Graph()
    G.add_nodes_from(trajectories)
    for i in range(len(trajectories)):
        for j in range(i, len(trajectories)):
            if i != j:
                if mutually_exclusive_trajectories(trajectories[i], trajectories[j]):
                    G.add_edge(trajectories[i], trajectories[j])
    return G

#Deprecated, remove?
def greedy_trajectory_algorithm(graph):
    optimal_trajectories = []
    plt.figure()
    while graph.number_of_nodes() != 0:
        nodes = list(graph.nodes)
        max_node = max(nodes , key= lambda n : n.value)
        _node_colors = []
        for i in range(len(nodes)):
            if nodes[i] == max_node:
                _node_colors.append('green')
            elif nodes[i] in graph.neighbors(max_node):
                _node_colors.append('red')
            else:
                _node_colors.append('blue')
        nx.draw(graph, with_labels=True, node_color=_node_colors)
        optimal_trajectories.append(max_node)
        for n in list(graph.neighbors(max_node)):
            graph.remove_node(n)
        graph.remove_node(max_node)
        plt.show()
        inn = print(input("Continue?"))
        if inn == 'n':
            return
    return optimal_trajectories


#Computes a pseudogreedy optimal set of trajectories, given a function to determine the next node to add.
def abstract_trajectory_algorithm(graph, choice_function,visualize=True):
    optimal_trajectories = []
    if visualize:
        plt.figure()
    while graph.number_of_nodes() != 0: #while there still are nodes left
        nodes = list(graph.nodes)
        chosen_node = choice_function(nodes) #choose most optimal node based on given choice function
        _node_colors = []
        for i in range(len(nodes)): #color nodes for visual purposes
            if nodes[i] == chosen_node:
                _node_colors.append('green')
            elif nodes[i] in graph.neighbors(chosen_node):
                _node_colors.append('red')
            else:
                _node_colors.append('blue')
        if visualize:
            nx.draw(graph, with_labels=True, node_color=_node_colors)
            plt.show()
        optimal_trajectories.append(chosen_node)
        for n in list(graph.neighbors(chosen_node)): #remove chosen node and neighbours, given that they are mutually exclusive
            graph.remove_node(n)
        graph.remove_node(chosen_node)
    print("Algorithm: " + choice_function.__name__ + ' sum: ' +str(sum(n.value for n in optimal_trajectories))) #print sum of trajectories
    return optimal_trajectories

#Choice function for pseudogreedy algorithm. Finds the best node by scaling all values by dividing with the sum of blocked nodes.
def weight_transformation(nodes):
    transformed_weights = []
    for i in range(len(nodes)):
        value_adjacent_nodes = 0
        for n in nodes[i].collisions:
            value_adjacent_nodes += n#.value
        if value_adjacent_nodes == 0:
            value_adjacent_nodes = 1
        transformed_weights.append(nodes[i].value /value_adjacent_nodes)
    
    return nodes[transformed_weights.index(max(transformed_weights))]

#Greedy algorithm choice function
def greedy(nodes):
    return  max(nodes, key= lambda n : n.value)

#Random choice function, used for benchmarks
def random_choice(nodes):
    return random.choice(nodes)

#Scale weights by dividing every node with the amount of nodes it blocks.
def NN_transformation(nodes):
    transformed_weights = []
    for i in range(len(nodes)):
        number_of_adjacent_nodes = 0
        for n in nodes[i].collisions:
            number_of_adjacent_nodes += 1
        if number_of_adjacent_nodes == 0:
            number_of_adjacent_nodes =1
        transformed_weights.append(nodes[i].value /number_of_adjacent_nodes)
    
    return nodes[transformed_weights.index(max(transformed_weights))]

#Helper function to check for collisions given an optimal trajectory list. Used for determining if the algorithms work correctly.
def check_for_collisions(optimal_trajectories):
    donors, targets, ids = []
    for t in optimal_trajectories:
        if t.donor in donors:
            return True
        if t.target in targets:
            return True
        elif t in ids:
            return True
        donors.append(t.donor)
        targets.append(t.target)
        ids += t.collisions
    return False
