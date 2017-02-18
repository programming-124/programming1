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

large_graph = 16384
max_weight = 0.00038189538245798719
num_nodes = 131072
num_trials = 1

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
    print "created graph"
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
        
        nodes_in_other_cut = list(set(range(0,n)) - set(T.keys()))
        
        for w in nodes_in_other_cut:
            try:
                weight_vw = graph[v][w]

                if dist[w] > weight_vw:
                    dist[w] = weight_vw
                    prev[w] = v
                    # replace later with distance of w
                    H[w] = dist[w]
            except:
                pass
    return T, sum(edge_weights)
            

"""
Get stats
"""
def get_average(num_trials, N):
    maxes = []
    weights = []
    for i in xrange(0, num_trials):
        graph = create_complete_graph(N)
        T, weight = prim(graph, N)
        weights.append(weight)
        max_el = max(T.values())
        maxes.append(max_el)
#         print "finished graph" + str(i)
    return max(maxes), mean(weights)


max_of_maxes, mean_weights = get_average(num_trials, num_nodes)
print (max_of_maxes, mean_weights)