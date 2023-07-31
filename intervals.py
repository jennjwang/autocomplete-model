import networkx as nx


def get_interval_borda(n, bg, source, radius, discount=1):
    sp = nx.shortest_path_length(bg, source=source, weight="edge_weight")
    interval = [0 for _ in range(n + 1)]
    for ballot, distance in sp.items():
        if distance <= radius:
            multiplier = bg.nodes()[ballot]["ballot_weight"] * (discount ** distance)
            ballot_length = len(ballot)
            for i in range(ballot_length):
                c = ballot[i]
                interval[c] += (1 - i/ballot_length)*multiplier
    total_weight = sum(interval)
    return [c / total_weight for c in interval]

