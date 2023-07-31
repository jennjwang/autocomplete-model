import pandas as pd
from parser import parse
import numpy as np
import matplotlib.pyplot as plt
# from randomwalk import sample_walk, cut_up_ballots, dedup, loop_erase, get_len
from collections import Counter
from itertools import product

election = {(3, 2, 1) : 1, (1, 2): 2, (3, 1, 2): 2, (1,): 3}

def closest_multiple(number, target):
    quotient = target // number
    lower_multiple = number * quotient
    upper_multiple = number * (quotient + 1)
    
    return lower_multiple if abs(target - lower_multiple) < abs(upper_multiple - target) else upper_multiple

def gen_transition_probs(election, n, is_normalized=True):

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
        fitted_length = closest_multiple(n, len(ranking))
        mod_ranking = (0, ) + ranking + (0, )
        padded = mod_ranking + (0, ) * (fitted_length - len(ranking))
        src = 0
        # targ = n
        while src <= len(ranking):
            src_node = padded[src:src+n]
            src += n
            targ_node = padded[src: src+n]
            print('ranking', ranking)
            print('src', src_node)
            print('targ', targ_node)
            if len(targ_node) == 0:
                break;
            src_ind = perms_ind[src_node]
            targ_ind = perms_ind[targ_node]
            mat[src_ind][targ_ind] += election[ranking]
    
# [[0. 5. 0. 3.]
#  [3. 0. 4. 0.]
#  [2. 1. 0. 0.]
#  [0. 2. 1. 0.]]


        # print('ranking', ranking)

        # top = padded[0:n]

        # print('top', top)
        # top_ind = perms_ind[top]

        # mat[top_ind] += election[ranking]

        # bottom = padded[-n:]

        # print('bottom', bottom)
        # # print('bottom_ind', bottom_ind)

        # bottom_ind = perms_ind[bottom]
        # print('bottom', bottom)
        # print('bottom_ind', bottom_ind)
        # print(ranking)
        # print(len(ranking))
        # print(len(cands))

        # mat[bottom_ind] += election[ranking]

    print(mat)

    # for row in mat:
    #     row_str = " ".join(f"{elem:>{nrows}}" for elem in row)
    #     print(row_str)

    # print(complete)

    # for r in range(1, nrows):
    #     for c in range(1, nrows):
    #         for ranking in all_rankings:
    #             if (r == c):
    #                 continue
    #             # print('ranking', ranking)
    #             # print('r', r)
    #             # print('c', c)
    #             if r in ranking and c in ranking:
    #                 r_index = ranking.index(r)
    #                 c_index = ranking.index(c)
    #                 # print('r_ind', r_index)
    #                 # print('c_ind', c_index)
    #                 if r_index == c_index - 1:
    #                     mat[r][c] += election[ranking]
    #                     # print('mat_val', mat[r][c])


    # if not is_normalized:
    #     return mat
    # norm_mat = row_normalize(mat)
    # return norm_mat

gen_transition_probs(election=election, n=2)

def sample_walk(mat, num_ballots, perm_inds):
    
    src_prob = ...

    # cands = range(len(mat))
    # src = 0
    # walk = []
    # walk.append(src)
    # b_count = 0
    # while b_count != num_ballots:
    #     targ_prob = mat[src]
    #     # print(targ_prob)
    #     targ = np.random.choice(a=cands, size=1, p=targ_prob)[0]
    #     # print(targ)
    #     walk.append(targ)
    #     if targ == 0:
    #         b_count  += 1
    #     # print(f'step {i}: {src} -> {targ} with p: {targ_prob[targ]}')
    #     src = targ
    
    # # print(walk)
    # return walk

