"""
Generators for BA homophilic network with minority and majority.
written by: Fariba Karimi
Date: 22-01-2016
Original version available at: https://github.com/gesiscss/homophilic_networks/blob/master/generate_homophilic_graph_symmetric.py
Edited by: Ekaterina Landgren
Date: 06-12-2024
"""

import networkx as nx
from collections import defaultdict
import random
import bisect
import copy
colors=['#19bdff','#f2d138']


def homophilic_ba_graph(N, m , minority_fraction, homophily,seed=None):
    """Return homophilic random graph using BA preferential attachment model.

    A graph of n nodes is grown by attaching new nodes each with m
    edges that are preferentially attached to existing nodes with high
    degree. The connections are established by linking probability which
    depends on the connectivity of sites and the homophily(similarities).
    homophily varies ranges from 0 to 1.

    Parameters
    ----------
    N : int
        Number of nodes
    m : int
        Number of edges to attach from a new node to existing nodes
    seed : int, optional
        Seed for random number generator (default=None).

    minority_fraction : float
        fraction of minorities in the network

    homophily: float
        value between 0 to 1. similarity between nodes. if nodes have same attribute
        their homophily (distance) is smaller.

    Returns
    -------
    G : Graph

    Notes
    -----
    The initialization is a graph with with m nodes and no edges.

    References
    ----------
    .. [1] A. L. Barabasi and R. Albert "Emergence of scaling in
       random networks", Science 286, pp 509-512, 1999.
    """

    G = nx.Graph()
    
    if seed!=None:
        random.seed(seed)

    minority = int(minority_fraction * N)

    minority_nodes = set(random.sample(range(N),minority))
    minority_mask = [node in minority_nodes for node in range(N)]
    G.add_nodes_from([(node, {"color": colors[1] if node in minority_nodes else colors[0]})\
        for node in range(N)])

    target_list=list(range(m))
    source = m #start with m nodes

    while source < N:
        targets = _pick_targets(G,source,target_list,minority_mask,homophily,m)

        if targets != set(): #if the node does  find the neighbor
            G.add_edges_from(zip([source]*m,targets))

        target_list.append(source)
        source += 1

    return G, list(minority_nodes)

def _pick_targets(G,source,target_list,minority_mask,homophily,m):

    target_prob_dict = {}
    for target in target_list:
        target_prob = homophily if minority_mask[source] == minority_mask[target] else 1 - homophily
        target_prob *= (G.degree(target)+0.00001)
        target_prob_dict[target] = target_prob

    prob_sum = sum(target_prob_dict.values())

    targets = set()
    target_list_copy = copy.copy(target_list)
    count_looking = 0
    if prob_sum == 0:
        return targets #it returns an empty set

    while len(targets) < m:
        count_looking += 1
        if count_looking > len(G): # if node fails to find target
            break
        rand_num = random.random()
        cumsum = 0.0
        for k in target_list_copy:
            cumsum += float(target_prob_dict[k]) / prob_sum
            if rand_num < cumsum:
                targets.add(k)
                target_list_copy.remove(k)
                break
    return targets


if __name__ == '__main__':
    graph = homophilic_ba_graph(N = 100, m = 2 , minority_fraction = 0.1, homophily= 1)