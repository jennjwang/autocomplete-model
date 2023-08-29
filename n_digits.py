import pandas as pd
from parser import parse
import numpy as np
import matplotlib.pyplot as plt
from randomwalk import sample_walk, cut_up_ballots, dedup, loop_erase, get_len
from collections import Counter
from itertools import product
from mcmc import row_normalize, gen_transition_probs
from collections import OrderedDict

# election = {(3, 2, 1): 1, (1, 2): 2, (3, 1, 2): 2, (1,): 3}
election = election, names, location = parse("Data/edinburgh17-16.blt")


def closest_multiple(number, target):
    quotient = target // number
    lower_multiple = number * quotient
    upper_multiple = number * (quotient + 1)

    return lower_multiple if abs(target - lower_multiple) < abs(upper_multiple - target) else upper_multiple


def gen_transition_probs_n(election, n):
    '''
    take a ranking
    take n digits in ranking
    find ind
    src = ind
    take next n digits in ranking
    find ind
    targ = ind
    rec[src][targ] = weight
    '''

    all_rankings = list(election.keys())

    cands = [item for ranking in all_rankings for item in ranking]
    cands = set(cands)
    cands.add(0)

    perms = list(product(cands, repeat=n))
    nrows = len(perms)

    perms_ind = {perm: index for index, perm in enumerate(perms)}

    print(perms_ind)

    mat = np.zeros((nrows, nrows))

    for ranking in all_rankings:
        mod_ranking = (0, ) + ranking + (0, )
        fitted_length = closest_multiple(n, len(mod_ranking))
        print(fitted_length)
        print(len(ranking))
        padded = mod_ranking + (0, ) * (fitted_length - len(ranking))
        src = 0
        while src <= len(ranking):
            src_node = padded[src:src+n]
            src += n
            targ_node = padded[src: src+n]
            targ_node = targ_node + (0, ) * (n - len(targ_node))
            print('ranking', ranking)
            print('src', src_node)
            print('targ', targ_node)
            if len(targ_node) == 0:
                break
            src_ind = perms_ind[src_node]
            targ_ind = perms_ind[targ_node]
            mat[src_ind][targ_ind] += election[ranking]
    return mat, perms_ind


def sample_walk_n(mat, num_ballots, perm_inds):

    nodes = list(perm_inds.keys())

    starting_nodes = {key: value for key,
                      value in perm_inds.items() if key[0] == 0}

    src_prob = {k: sum(mat[v]) for k, v in starting_nodes.items()}
    normalized_src_prob = [src_prob[k] / sum(list(src_prob.values()))
                           for k in src_prob.keys()]

    src_ind = np.random.choice(a=list(starting_nodes.values()),
                               p=normalized_src_prob, size=1)[0]

    perms = list(starting_nodes.keys())
    # print(perms[src])

    walk = []
    ballot = []
    ballot.append(perms[src_ind])
    b_count = 0

    while b_count != num_ballots:

        # print(list(mat[src_ind]))

        if sum(list(mat[src_ind])) == 0 or nodes[src_ind][1] == 0:
            b_count += 1
            # ballot.append(nodes[src_ind])
            walk.append(ballot)
            ballot = []
            src_ind = np.random.choice(a=list(starting_nodes.values()),
                                       p=normalized_src_prob, size=1)[0]
            ballot.append(perms[src_ind])

        targ_probs = [i / sum(list(mat[src_ind])) for i in list(mat[src_ind])]

        # print(targ_probs)
        targ_ind = np.random.choice(a=len(targ_probs), p=targ_probs)
        # print(targ_ind)

        # print(nodes[targ_ind])
        ballot.append(nodes[targ_ind])

        src_ind = targ_ind

    walk.append(ballot)
    return walk


def n_parse_walk(ballots, n):
    res = []
    for b in ballots:
        ballot = []
        for elm in b:
            # if sum(elm) == 0:
            #     ballot.append(0)
            # else:
            for i in range(n):
                ballot.append(elm[i])
        ballot = [num for num in ballot if num != 0]
        # print(ballot)
        res.append(ballot)
    return res
    ...


if __name__ == "__main__":

    # mat_n3, perm_ind = gen_transition_probs_n(election=election, n=3)
    # ballots_n3 = sample_walk_n(mat_n3, 3, perm_inds=perm_ind)
    # print('ballot', ballots_n3)
    # walk = n_parse_walk(ballots=ballots_n3, n=3)
    # print('walk', walk)

    all_rankings = list(election.keys())
    cands = [item for ranking in all_rankings for item in ranking]
    cands = set(cands)

    n = sum(list(election.values()))
    print('num', n)
    mat_n1 = gen_transition_probs(
        election=election, is_normalized=True)
    walk = sample_walk(mat_n1, n)
    ballots_n1 = cut_up_ballots(walk)
    acc_ballots = [[list(key)] * value for key, value in election.items()]
    acc_ballots = [item for sublist1 in acc_ballots for item in sublist1]

    mat_n2, perm_ind = gen_transition_probs_n(election=election, n=2)
    ballots_n2 = sample_walk_n(mat_n2, n, perm_inds=perm_ind)
    ballots_n2 = n_parse_walk(ballots=ballots_n2, n=2)

    mat_n3, perm_ind = gen_transition_probs_n(election=election, n=3)
    print('started walk')
    walk_n3 = sample_walk_n(mat_n3, n, perm_inds=perm_ind)
    print('done walk')
    ballots_n3 = n_parse_walk(ballots=walk_n3, n=3)

    ballot_len_acc = Counter(get_len(acc_ballots))
    ballot_len_syn_n2 = Counter(get_len(ballots_n2))
    ballot_len_syn_n1 = Counter(get_len(ballots_n1))
    ballot_len_syn_n3 = Counter(get_len(ballots_n3))

    overvotes = {key: value for key,
                 value in ballot_len_syn_n1.items() if key > len(cands)}
    for key in overvotes.keys():
        ballot_len_syn_n1[len(cands)] += overvotes[key]

    overvotes2 = {key: value for key,
                  value in ballot_len_syn_n2.items() if key > len(cands)}
    # print(overvotes2)

    for key in overvotes2.keys():
        ballot_len_syn_n2[len(cands)] += overvotes2[key]

    overvotes3 = {key: value for key,
                  value in ballot_len_syn_n3.items() if key > len(cands)}
    # print(overvotes3)

    for key in overvotes3.keys():
        ballot_len_syn_n3[len(cands)] += overvotes3[key]

    print('syn', ballot_len_syn_n3)

    dict_list = [dict(ballot_len_acc), dict(
        ballot_len_syn_n2), dict(ballot_len_syn_n1), dict(ballot_len_syn_n3)]

    # Extract items and frequencies from dictionaries
    items = list(ballot_len_acc.keys())
    items.sort()  # Sort the items alphabetically
    frequencies = np.array([[d[item] for item in items] for d in dict_list])

    # Create figure and axis objects
    fig, ax = plt.subplots()

    # Set the width of each bar
    bar_width = 0.2

    # Set the positions of the bars on the x-axis
    bar_positions = np.arange(len(items))

    # Plot the grouped bar chart
    for i in range(len(dict_list)):
        ax.bar(bar_positions + i * bar_width, frequencies[i], width=bar_width)

    # Set the x-axis labels
    ax.set_xticks(bar_positions + (len(dict_list) - 1) * bar_width / 2)
    ax.set_xticklabels(items)

    # Set the labels and title
    ax.set_xlabel('Ballot Length')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Ballot Lengths')

    ax.legend(['actual', 'synthetic (n=2)',
              'synthetic (n=1)', 'synthetic (n=3)'])

    plt.show()
