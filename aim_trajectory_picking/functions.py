import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt


class Trajectory:
    def __init__(self,_id, _donor, _target, _value, _risk=0):
        self.id = _id
        self.donor = _donor
        self.target = _target
        self.value = _value
        self.risk = _risk
        self.collisions = []
    
    def add_collision(self, trajectory):
        if trajectory.id not in self.collisions:
            self.collisions.append(trajectory.id)
            trajectory.add_collision(self)

    def add_collision_by_id(self, id):
        self.collisions.append(id)
 
    def __str__(self):
        return str(self.id) + ": "+ self.donor + "-->" + self.target + "  Value " + str(self.value) + " "
    
    def __eq__(self, other):
        if not isinstance(other, Trajectory):
            # don't attempt to compare against unrelated types
            return NotImplemented
        return self.id == other.id and self.donor == other.donor and self.target == other.target and self.value == other.value and self.collisions == other.collisions

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
        trajectories.append(Trajectory(i, random.choice(donors), random.choice(targets),random.randint(0,10))) #a trejectory is remembered by a tuple of start and end
    
    for i in range(no_trajectories):
        for j in range(i, no_trajectories):
            if i !=j:
                if np.random.binomial(1,collision_rate):
                    #print("added trajectory for " + str(i) + " and" + str(j))
                    trajectories[i].add_collision(trajectories[j])
                    trajectories[j].add_collision(trajectories[i])
    
    
    return donors, targets, trajectories

def bipartite_graph(donors, targets, trajectories):
    g = nx.Graph()
    g.add_nodes_from(donors+ targets)
    for t in trajectories:
        g.add_edge(t.donor, t.target, weight=t.value)
    _node_color =[]
    for i in range(len(donors)):
        _node_color.append('green')
        
    for i in range(len(targets)):
        _node_color.append('red')
        
    nx.draw(g, nx.bipartite_layout(g,donors), node_color=_node_color, with_labels=True)
    
def mutually_exclusive_trajectories(t1, t2):
    if(t1.donor == t2.donor):
        return True
    if(t1.target == t2.target):
        return True
    if(t2.id in t1.collisions):
        return True
    return False


def transform_graph(trajectories):
    G = nx.Graph()
    G.add_nodes_from(trajectories)
    for i in range(len(trajectories)):
        for j in range(i, len(trajectories)):
            if i != j:
                if mutually_exclusive_trajectories(trajectories[i], trajectories[j]):
                    G.add_edge(trajectories[i], trajectories[j])
    return G

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

def abstract_trajectory_algorithm(graph, choice_function,visualize=True):
    optimal_trajectories = []
    plt.figure()
    while graph.number_of_nodes() != 0:
        nodes = list(graph.nodes)
        chosen_node = choice_function(nodes)
        _node_colors = []
        for i in range(len(nodes)):
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
        for n in list(graph.neighbors(chosen_node)):
            graph.remove_node(n)
        graph.remove_node(chosen_node)
    print(sum(n.value for n in optimal_trajectories))
    return optimal_trajectories

def weight_transformation(nodes):
    transformed_weights = []
    for i in range(len(nodes)):
        value_adjacent_nodes = 0
        for n in nodes[i].collisions:
            value_adjacent_nodes += n.value
        if value_adjacent_nodes == 0:
            value_adjacent_nodes = 1
        transformed_weights.append(nodes[i].value /value_adjacent_nodes)
    
    return nodes[transformed_weights.index(max(transformed_weights))]

def greedy(nodes):
    return  max(nodes, key= lambda n : n.value)

def random_choice(nodes):
    return random.choice(nodes)

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