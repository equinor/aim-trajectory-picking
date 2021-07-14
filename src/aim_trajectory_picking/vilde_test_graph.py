import igraph as ig
import random
import numpy as np
import matplotlib.pyplot as plt
import time
from itertools import combinations
from aim_trajectory_picking import functions as func
import time
import JSON_IO
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

def bipartite_graph_igraph(donors, targets, trajectories):
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

    Returns:
    --------
        g: nx.Graph()
            a bipartite graph with the donors + targets as nodes and trajectories as edges
    '''
    g = ig.Graph()
    g.add_vertices(donors + targets)
    for t in trajectories:
        g.add_edges([(t.donor, t.target)])
    g.es["weight"]=[t.value for value in trajectories]
    _node_color =[]
    for i in range(len(donors)):
        _node_color.append('green')
    for i in range(len(targets)):
        _node_color.append('red')
    '''
    if visual:
        plt.figure()
        pos = nx.bipartite_layout(g, donors) # Not a function 
        nx.draw(g, pos, node_color=_node_color, with_labels=True) # Not a function 
        labels = nx.get_edge_attributes( g,'weight')  # Not a function 
        nx.draw_networkx_edge_labels(g,pos,edge_labels=labels)  #g.es to add labels: g.es["Edge"] = [list of edges]
        plt.show() 
    '''
    return g


donors, targets, trajectories = func.create_data(10, 10, 100, 0.05)

#g1 = bipartite_graph_igraph(donors, targets, trajectories)
def transform_graph_igraph(trajectories):
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
    G = ig.Graph()
    G.add_vertices(len(trajectories))
    #start = time.perf_counter()
    G.vs["value"]=[t.value for t in trajectories]
    G.vs["donor"]=[t.donor for t in trajectories]
    G.vs["target"]=[t.target for t in trajectories]
    G.vs["name"]=[t.id for t in trajectories]
    #stop = time.perf_counter()
    #difference = stop-start
    #print("timer list comprehension", difference)
    collisions = []
    for i in range(len(trajectories)):
        for j in range(i, len(trajectories)):
            if i != j:
                if func.mutually_exclusive_trajectories(trajectories[i], trajectories[j]):
                    collisions.append((trajectories[i].id, trajectories[j].id))
    G.add_edges(collisions)
    return G

def greedy_igraph(trajectories,*,visualize=False):
    g = transform_graph_igraph(trajectories)
    optimal_trajectories = []
    while len(g.vs) != 0:
        max_node = max(g.vs, key= lambda n: n['value'])
        optimal_trajectories.append(trajectories[max_node['name']])
        g.delete_vertices(max_node.neighbors() + [max_node])
    dictionary = {}
    dictionary['value'] = sum(n.value for n in optimal_trajectories)
    dictionary['trajectories'] = optimal_trajectories
    return dictionary

def greedy_igraph_from_graph(g):
    optimal_trajectories = []
    while len(g.vs) != 0:
        max_node = max(g.vs, key= lambda n: n['value'])
        optimal_trajectories.append(trajectories[max_node['name']])
        g.delete_vertices(max_node.neighbors() + [max_node])
    dictionary = {}
    dictionary['value'] = sum(n.value for n in optimal_trajectories)
    dictionary['trajectories'] = optimal_trajectories
    return dictionary

def remove_collisions_bipartite_matching(trajectories,*,visualize=False):
    G = ig.Graph()
    G.add_vertices(len(trajectories))
    #start = time.perf_counter()
    G.vs["value"]=[t.value for t in trajectories]
    G.vs["donor"]=[t.donor for t in trajectories]
    G.vs["target"]=[t.target for t in trajectories]
    G.vs["name"]=[t.id for t in trajectories]
    collisions = []
    for i in range(len(trajectories)):
        for j in range(i, len(trajectories)):
            if i != j:
                if trajectories[i].id in  trajectories[j].collisions:
                    collisions.append((trajectories[i].id, trajectories[j].id))
    G.add_edges(collisions)
    optimal_trajectories_for_bip_matching = greedy_igraph_from_graph(G)
    donors, targets = func.get_donors_and_targets_from_trajectories(optimal_trajectories_for_bip_matching['trajectories'])
    vertexes = donors + targets
    graph = ig.Graph()
    graph.add_vertices(len(vertexes))
    graph.vs['name'] = donors + targets
    graph.vs['type'] = [True if s[0] == 'T' else False for s in graph.vs['name'] ]
    graph.add_edges([(t.donor , t.target) for t in optimal_trajectories_for_bip_matching['trajectories']])
    graph.es['value'] = [t.value for t in optimal_trajectories_for_bip_matching['trajectories']]
    graph.es['name'] = [t.id for t in optimal_trajectories_for_bip_matching['trajectories']]
    #print(graph.is_bipartite())
    #bi_graph = ig.Graph.Bipartite( donors + targets, [(t.donor , t.target) for t in optimal_trajectories_for_bip_matching], weights='value')
    matching = graph.maximum_bipartite_matching(types='type', weights='value', eps=0.01)
    
    optimal_trajectories =  [trajectories[i] for i in matching.edges()['name']]
    return func.optimal_trajectories_to_return_dictionary(optimal_trajectories)

def igraph_invert_and_clique(trajectories,*,visualize=False):
    g = transform_graph_igraph(trajectories)
    inverter = g.complementer(loops=False)
    #print([trajectories[i] for i in inverter.largest_cliques()[0]])
    return func.optimal_trajectories_to_return_dictionary([trajectories[i] for i in inverter.largest_cliques()[0]])


#trajectories = JSON_IO.read_trajectory_from_json(r'even_datasets\even_test_2.txt')
_,_,trajectories = func.create_data(10,10,100, 0.05)
start = time.perf_counter()
#graph = func.transform_graph(trajectories)
# for node in graph.nodes():
#     print(node.id)
answer = func.greedy_algorithm(trajectories)
print("nx value: " + str(answer['value']))
stop = time.perf_counter()
print("networkx", stop-start)

start1 = time.perf_counter()
#graph1 = transform_graph_igraph(trajectories)
answer1 = greedy_igraph(trajectories)
# for node in graph1.vs:
#     print(node['name'])
print("igraph value:" + str(answer1['value']))
stop1 = time.perf_counter()
print("igraph", stop1-start1)

start_bip = time.perf_counter()
exact = igraph_invert_and_clique(trajectories)
end_bip = time.perf_counter()
func.check_for_collisions(exact['trajectories'])
print("value exact: ", exact['value'])
print("time exact sol with igraph:", end_bip - start_bip)

start_bip = time.perf_counter()
exact = func.invert_and_clique(trajectories)
end_bip = time.perf_counter()
func.check_for_collisions(exact['trajectories'])
print("value exact: ", exact['value'])
print("time exact sol with nx:", end_bip - start_bip)

start_bip = time.perf_counter()
answer_bip = remove_collisions_bipartite_matching(trajectories)
end_bip = time.perf_counter()
print("Igraph bip value:" + str(answer_bip['value']))
print("igraph bip time", end_bip-start_bip)
func.check_for_collisions(answer_bip['trajectories'])

start_bip = time.perf_counter()
answer_bip = func.bipartite_matching_removed_collisions(trajectories, False)
end_bip = time.perf_counter()
print("Nx bip value:" + str(answer_bip['value']))
print("NX bip time", end_bip-start_bip)