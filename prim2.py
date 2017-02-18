import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)
import collections
from collections import Counter

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)
"""
Constants
"""

large_graph = 4096
max_weight = 0.020971164721624622
num_nodes = 65536
num_trials = 1
dimensions = 2

"""
create_complete_graph(n) 
"""

def create_complete_graph(n):
    vs = list(range(0, n))
    graph = collections.defaultdict(dict)
    for u in xrange(0, n):
        weights = np.random.random_sample((1,n))[0]
        for w in xrange(u+1,n):
            if n >= large_graph:
                if weights[w] < max_weight:
                    graph[u][w] = weights[w]
                    graph[w][u] = weights[w]
            else:
                graph[u][w] = weights[w]
                graph[w][u] = weights[w]
    return graph

# for a complete graph, m = O(n^2)
# create a graph, matrix representation
"""
multiple dimensions
"""

def create_complete_graph_dimensional(n, d):
    vs = list(range(0, n))
    graph = collections.defaultdict(dict)
    points = np.random.random_sample((n,d))
    for u in xrange(0, n):
        for w in xrange(u+1,n):
            weight = np.linalg.norm(points[u]-points[w])
            if n >= large_graph:
                if weight < max_weight:
                    graph[u][w] = weight
                    graph[w][u] = weight
            else:
                graph[u][w] = weight
                graph[w][u] = weight
    return graph

"""
Delete min dict
input: {s:0}
deletes smallest priority element
"""
def delete_min_dict(H):
    v = min(H, key=H.get)
    edge_weight = H[v]
    del H[v]
    return v, edge_weight, H

"""
Prim's algorithm
takes a graph of size n
returns a tree
"""
"""
Prim's algorithm
takes a graph of size n
returns a tree

Look through each edge instead of iterating through all the edges
"""
def prim(graph, n):
    # keep 
    dist= collections.defaultdict(int)
    # keep track of previous node for each vertex v
    prev = collections.defaultdict(int)
    vs = list(range(0, n))
    for v in vs:
        dist[v] = float("inf")
    s = 0
    dist[s] = 0

    # for now, heap to keep track of what to add is just an array
    H = {s:0}
    T = {}
    edge_weights = []
    while H != {}:
        v, edge_weight, H = delete_min_dict(H)
        T[v] = edge_weight
        edge_weights.append(edge_weight)
        # loop through all the edges of graph (v,w) not already connected in the current tree
        
        for w in graph[v]:
            # only care about edge to w if w not in our T, now can access in constant time to check if in other cut
            if w not in T:
                weight_vw = graph[v][w]

                if dist[w] > weight_vw:
                    dist[w] = weight_vw
                    prev[w] = v
                    # replace later with distance of w
                    H[w] = dist[w]
    return T, sum(edge_weights)
            
def create_graph(N, dimensions):
    if dimensions == 1:
        graph = create_complete_graph(N)
        print "created graph"
        return graph
    graph = create_complete_graph_dimensional(N, dimensions)
    print "created graph"
    return graph

"""
Get stats
"""
def get_average(num_trials, N, dimensions):
    maxes = []
    weights = []
    for i in xrange(0, num_trials):
        graph = create_graph(N, dimensions)
        T, weight = prim(graph, N)
        weights.append(weight)
        print ("weight", weight)
        max_el = max(T.values())
        print ("max_el", max_el)
        maxes.append(max_el)
        print "finished with graph" + str(i)
    return max(maxes), mean(weights)

print ("num_trials", num_trials, "num_nodes", num_nodes, "dimensions", dimensions)
max_of_maxes, mean_weights = get_average(num_trials, num_nodes, dimensions)
print (max_of_maxes, mean_weights)