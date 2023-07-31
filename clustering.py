import itertools
from clustering import *
import networkx as nx
import random
from fractions import Fraction
import pickle
import matplotlib.pyplot as plt


"""
TODO
Debug why plots are zero
Interval-aware clustering
Redo test 16 with new clustering function
"""


def draw(bg):
    nx.draw(bg, with_labels=True, labels={
            node: "".join(map(str, node)) + "\n" + str(m["ballot_weight"])
            for node, m in bg.nodes(data=True)})
    plt.show()


def ballot_graph(n, k, election=None, remove_k_minus_1=False):
    bg = nx.Graph()
    for num_positions_filled in range(1, k + 1):
        if remove_k_minus_1 and num_positions_filled == k - 1:
            continue
        for ballot in itertools.permutations(range(1, n + 1), num_positions_filled):
            bg.add_node(ballot, ballot_weight=0 if election else 1)
    if election:
        for ballot, ballot_weight in election.items():
            truncated_ballot = ballot[:k]
            if truncated_ballot in bg.nodes():
                bg.add_node(truncated_ballot, ballot_weight=ballot_weight
                            + bg.nodes()[truncated_ballot]["ballot_weight"])
            else:
                raise Exception(f"Unknown ballot found in election: {truncated_ballot}")
    for old_ballot in bg.nodes():
        num_positions_filled = len(old_ballot)
        if num_positions_filled > 1 and not (remove_k_minus_1 and num_positions_filled == k - 1):
            for i in range(num_positions_filled - 1):
                # Add swapping edges.
                new_ballot = list(old_ballot)
                new_ballot[i + 1] = old_ballot[i]
                new_ballot[i] = old_ballot[i + 1]
                bg.add_edge(old_ballot, tuple(new_ballot), edge_weight=1)
            # Add truncation edges.
            if len(old_ballot) == k and remove_k_minus_1:
                bg.add_edge(old_ballot, old_ballot[:-2], edge_weight=1)
            else:
                bg.add_edge(old_ballot, old_ballot[:-1], edge_weight=1)
    return bg


def pickle_distance_matrix(n, k):
    bg = ballot_graph(n, k)
    d = nx.floyd_warshall(bg, weight="edge_weight")
    with open(f"ballot_graph_{n}_{k}", 'wb') as f:
        pickle.dump({v1: {v2: d[v1][v2] for v2 in bg.nodes()} for v1 in bg.nodes()}, f)


def k_means(bg, n, trunc, k, num_repetitions=None):
    if num_repetitions:
        to_return = {}
        for _ in range(num_repetitions):
            centers = frozenset(k_means(bg, n, trunc, k))
            if centers in to_return:
                to_return[centers] += 1
            else:
                to_return[centers] = 1
        return to_return
    with open(f"ballot_graph_{n}_{trunc}", 'rb') as f:
        d = pickle.load(f)
    old_centers = {}
    new_centers = set(random.sample(bg.nodes(), k=k))
    num_iterations = 0
    while old_centers != new_centers:
        old_centers = new_centers
        new_centers = set()
        num_iterations += 1
        #print(num_iterations)
        
        # Compute clusters, dividing equally when there are ties.
        weight_contribution_to_cluster = {center: {} for center in old_centers}
        for node, node_data in bg.nodes(data=True):
            min_distance = min(d[node][center] for center in old_centers)
            divide_among = [center for center in old_centers if d[node][center] == min_distance]
            num_to_divide_among = len(divide_among)
            for center in divide_among:
                weight_contribution_to_cluster[center][node] = \
                        Fraction(node_data["ballot_weight"], num_to_divide_among)
        
        # Define new centers.
        for old_center in old_centers:
            w = weight_contribution_to_cluster[old_center]
            sum_of_distances = {node: 0 for node in bg.nodes()}
            for node in bg.nodes():
                for node_in_cluster, weight_contribution in w.items():
                    sum_of_distances[node] += weight_contribution*d[node][node_in_cluster]
            sorted_nodes = list(bg.nodes())
            random.shuffle(sorted_nodes)
            sorted_nodes = sorted(sorted_nodes, key=lambda node: sum_of_distances[node])
            i = 0
            while sorted_nodes[i] in new_centers:
                i += 1
            new_centers.add(sorted_nodes[i])
    print(f"Convergence after {num_iterations} iterations.")
    return new_centers


def annotate_with_intervals(bg, interval):
    """Directed weighted distances based on Mala's drawing."""
    bdg = bg.to_directed()
    for b1, b2 in bg.edges():
        l1 = len(b1)
        l2 = len(b2)
        if l1 == l2:
            for i in range(l1):
                if b1[i] != b2[i]:
                    break
            w_forward = (interval[b1[i]]/interval[b2[i]])/(2**(i + 1))
            w_backward = (interval[b2[i]]/interval[b1[i]])/(2**(i + 1))
        elif l1 > l2:
            w_forward = w_backward = interval[b2[-1]]/(2**l1)
        else:
            w_forward = w_backward = interval[b1[-1]]/(2**l2)
        bdg.add_edge(b1, b2, edge_weight_using_intervals=w_forward)
        bdg.add_edge(b2, b1, edge_weight_using_intervals=w_backward)
    return bdg


def compute_weighted_distance(bdg, a, b):
    return nx.shortest_path_length(bdg, source=a, target=b, weight="edge_weight_using_intervals")


def interval_aware_clustering(bg, n, trunc, k, discount, num_repetitions=None):
    if num_repetitions:
        to_return = {}
        for _ in range(num_repetitions):
            intervals = frozenset(interval_aware_clustering(bg, n, trunc, k, discount))
            if intervals in to_return:
                to_return[intervals] += 1
            else:
                to_return[intervals] = 1
        return to_return
    with open(f"ballot_graph_{n}_{trunc}", 'rb') as f:
        d = pickle.load(f)
    old_centers = []
    new_centers = list(random.sample(bg.nodes(), k=k))
    new_intervals = [[0] + [1/n for __ in range(n)] for _ in range(k)]
    num_iterations = 0
    while old_centers != new_centers:
        old_centers = new_centers
        new_centers = []
        old_intervals = new_intervals
        bdgs = [annotate_with_intervals(bg, interval) for interval in new_intervals]
        new_intervals = []
        num_iterations += 1
        
        # Compute clusters based on centers and intervals, dividing equally when there are ties.
        weight_contribution_to_cluster = {center: {} for center in old_centers}
        for node, node_data in bg.nodes(data=True):
            weighted_distance = {old_centers[i]: compute_weighted_distance( \
                                                 bdgs[i], old_centers[i], node \
                                                 ) for i in range(k)}
            min_distance = min(weighted_distance[old_centers[i]] for i in range(k))
            divide_among = [center for center in old_centers if weighted_distance[center] \
                            == min_distance]
            num_to_divide_among = len(divide_among)
            for center in divide_among:
                weight_contribution_to_cluster[center][node] = \
                        Fraction(node_data["ballot_weight"], num_to_divide_among)
        
        # Define new centers, same as in k_means.
        for old_center in old_centers:
            w = weight_contribution_to_cluster[old_center]
            sum_of_distances = {node: 0 for node in bg.nodes()}
            for node in bg.nodes():
                for node_in_cluster, weight_contribution in w.items():
                    sum_of_distances[node] += weight_contribution*d[node][node_in_cluster]
            sorted_nodes = list(bg.nodes())
            random.shuffle(sorted_nodes)
            sorted_nodes = sorted(sorted_nodes, key=lambda node: sum_of_distances[node])
            i = 0
            while sorted_nodes[i] in new_centers:
                i += 1
            new_centers.append(sorted_nodes[i])
        
        # Define new intervals from centers.
        for center in new_centers:
            sp = nx.shortest_path_length(bg, source=center, weight="edge_weight")
            interval = [0 for _ in range(n + 1)]
            for ballot, distance in sp.items():
                multiplier = bg.nodes()[ballot]["ballot_weight"] * (discount ** distance)
                ballot_length = len(ballot)
                for i in range(ballot_length):
                    c = ballot[i]
                    interval[c] += (1 - i/ballot_length)*multiplier
            total_weight = sum(interval)
            new_intervals.append([c / total_weight for c in interval])
        
    print(f"Convergence after {num_iterations} iterations.")
    return tuple(map(tuple, new_intervals))

