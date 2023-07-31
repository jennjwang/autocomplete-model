import pandas as pd
from parser import parse
import numpy as np
import matplotlib.pyplot as plt
from randomwalk import sample_walk, cut_up_ballots, dedup
from collections import Counter

election, names, location = parse("Data/edinburgh17-16.blt")
print(len(election))
# print('election', election)
# print('names', names)
# print('loc', location)

# election = {(3, 2, 1) : 1, (1, 2): 2, (3, 1, 2): 2}

all_rankings = list(election.keys())
cands = [item for ranking in all_rankings for item in ranking]
cands = set(cands)
nrows = len(cands) + 1

def gen_transition_probs(election, is_normalized=True):

    all_rankings = list(election.keys())
    cands = [item for ranking in all_rankings for item in ranking]
    cands = set(cands)

    nrows = len(cands) + 1
    mat = np.zeros((nrows, nrows))

    for ranking in all_rankings:
        # print('ranking', ranking)
        top = ranking[0]
        # print('top', top)
        mat[0][top] += election[ranking]
        bottom = ranking[-1]
        # print('bottom', bottom)
        mat[bottom][0] += election[ranking]

    # for row in mat:
    #     row_str = " ".join(f"{elem:>{nrows}}" for elem in row)
    #     print(row_str)

    for r in range(1, nrows):
        for c in range(1, nrows):
            for ranking in all_rankings:
                if (r == c):
                    continue
                # print('ranking', ranking)
                # print('r', r)
                # print('c', c)
                if r in ranking and c in ranking:
                    r_index = ranking.index(r)
                    c_index = ranking.index(c)
                    # print('r_ind', r_index)
                    # print('c_ind', c_index)
                    if r_index == c_index - 1:
                        mat[r][c] += election[ranking]
                        # print('mat_val', mat[r][c])


    if not is_normalized:
        return mat
    norm_mat = row_normalize(mat)
    return norm_mat


def row_normalize(matrix):
    row_sums = matrix.sum(axis=1, keepdims=True)  # Calculate the row sums
    normalized_matrix = matrix / row_sums       # Perform element-wise division
    return normalized_matrix


# print('normalized...')
# for row in norm_mat:
#     row_str = " ".join(f"{elem:>{nrows}}" for elem in row)
#     print(row_str)


def stable_state(norm_mat):
    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(norm_mat.T)
    # print('eigens', eigenvectors.real)
    # print('eigenvectors', eigenvectors.real)

    # Find the eigenvector corresponding to eigenvalue of 1
    index_eigenvalue_1 = np.where(np.isclose(eigenvalues, 1))[0][0]
    eigenvector_1 = eigenvectors[:, index_eigenvalue_1]

    # print("Eigenvalue:", eigenvalues[index_eigenvalue_1].real)

    ss = eigenvector_1.real / sum(eigenvector_1.real)
    print("Stable state")
    print(ss)

    return ss

# print('steady state')
# print((np.abs(eigenvector_1).real))

# res = norm_mat.T
# n = 100
# for _ in range(n - 1):
#     res = res * norm_mat

# res = res * np.array([1, 0, 0, 0])
# print(res)
# print('normalized')
# print(norm_mat.T)

# M = norm_mat.T
# v = np.array([1, 0, 0, 0])

# print(sum(np.linalg.matrix_power(M, 100)@v))
# print(np.linalg.matrix_power(M, 100)@v)

# for n in range(1,30):
#     print(n,'th step, normalized:',np.around(M**n@v, decimals=5))

def gen_mentions(cands, election, is_normalized=True):
    mentions = {}
    for ranking in election:
        for cand in ranking:
            if cand in mentions:
                mentions[cand] += election[ranking]
            else:
                mentions[cand] = election[ranking]

    # print(mentions)
    mentions_prob = [mentions[cand] for cand in cands]
    print('Mentions (normalized)')
    norm = np.array(mentions_prob)  / sum(mentions_prob)
    print(norm)
    if not is_normalized:
        return mentions_prob
    return norm


def borda_from_election(cands, election, is_normalized=True):
    borda = {}
    n = len(cands)
    for ranking in election:
        weight = election[ranking]
        # print(ranking)
        for i in range(len(ranking)):
            cand = ranking[i]
            # print('cand', cand)
            positional_weight = n - i
            if cand in borda:
                borda[cand] += positional_weight * weight
            else:
                borda[cand] = positional_weight * weight
    print('Borda (normalized)')
    # print(borda)
    borda_prob = [borda[cand] for cand in cands]
    norm = np.array(borda_prob)  / sum(borda_prob)
    print(norm)
    if not is_normalized:
        return borda_prob
    return norm


def borda_from_mat(mat):
    nrows = len(mat)
    v = np.zeros((nrows, 1))
    v[0, 0] = 1
    # print('v')
    # print(v)

    q_mat = mat.copy()
    q_mat[:, 0] = 0
    q_mat[0, 0] = 1

    # print('q')
    # print(q_mat)

    p_mat = mat.copy()
    p_mat = p_mat

    # print('p')
    # print(p_mat)

    # print('p_mat', p_mat.shape)
    # print('q_mat', q_mat.shape)
    # print('v', v.shape)

    n = len(mat) - 1

    q_exp = np.zeros(q_mat.shape)

    for i in range(0, n):
        w = n - i
        # print('power', i)
        # print('weight', w)
        q_exp = q_exp + w * np.linalg.matrix_power(q_mat,i)

    # q_factor = sum([(n - i) * np.power(q_mat,i) for i in range(0, n)])
    res = np.matmul(q_exp, p_mat) @ v

    res = res.flatten()
    print((res[1:] / sum(res[1:])))
    return res[1:] / sum(res[1:])

if __name__ == "__main__":
    transition_mat = gen_transition_probs(election=election)
    be = borda_from_election(cands, election)
    it_mat = transition_mat.T
    ss = stable_state(transition_mat)
    ss = ss[1:] / sum(ss[1:])
    bm = borda_from_mat(it_mat)
    men = gen_mentions(cands, election)

    # mat = gen_transition_probs(election=election)
    n = 1390
    walk = sample_walk(transition_mat, n)
    ballots = cut_up_ballots(walk)
    deduped = list(map(dedup, ballots))

    syn_election = dict(Counter(tuple(lst) for lst in deduped))
    # syn_transition_mat = gen_transition_probs(election=syn_election)
    # syn_it_mat = syn_transition_mat.T
    syn_be = borda_from_election(cands, syn_election)
    syn_men = gen_mentions(cands, election=syn_election)

    cands = np.arange(1, len(transition_mat))
    print(cands)
    bar_width = 0.25
    plt.bar(cands - bar_width, men, bar_width,label='actual mentions', color='blue')
    plt.bar(cands + bar_width, ss, bar_width, label='stable state', color='orange')
    plt.bar(cands, syn_men, bar_width,label='synthetic (dedup) mentions', color='green')

    # Add labels and title
    plt.xlabel('candidates')
    plt.ylabel('values')
    plt.title('Mentions / SS for Scottish Election')
    plt.legend()
    plt.show()

# for row in it_mat:
#     row_str = " ".join(f"{elem:>{nrows}}" for elem in row)
#     print(row_str)
'''
mentions = {}
for ranking in election:
    for cand in ranking:
        if cand in mentions:
            mentions[cand] += election[ranking]
        else:
            mentions[cand] = election[ranking]

# print(mentions)
mentions_prob = [mentions[cand] for cand in cands]
print('Mentions (normalized)')
print(np.array(mentions_prob)  / sum(mentions_prob))

borda = {}
n = len(cands)
for ranking in election:
    weight = election[ranking]
    # print(ranking)
    for i in range(len(ranking)):
        cand = ranking[i]
        # print('cand', cand)
        positional_weight = n - i
        # print('positional', positional_weight)
        # print('weight', weight)
        if cand in borda:
            borda[cand] += positional_weight * weight
        else:
            borda[cand] = positional_weight * weight

print('Borda (normalized)')
# print(borda)
borda_prob = [borda[cand] for cand in cands]
print(np.array(borda_prob)  / sum(borda_prob))

'''