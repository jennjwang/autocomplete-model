import numpy as np
# from mcmc import gen_transition_probs
from parser import parse
from collections import OrderedDict

def sample_walk(mat, num_ballots):
    cands = range(len(mat))
    src = 0
    walk = []
    walk.append(src)
    b_count = 0
    while b_count != num_ballots:
        targ_prob = mat[src]
        print(targ_prob)
        targ = np.random.choice(a=cands, size=1, p=targ_prob)[0]
        # print(targ)
        walk.append(targ)
        if targ == 0:
            b_count  += 1
        # print(f'step {i}: {src} -> {targ} with p: {targ_prob[targ]}')
        src = targ
    
    print(walk)
    return walk

# mat = [[0.,0.4,0.,0.6],
#        [0.2,0.,0.8,0.],
#        [0.8,0.2,0.,0.],
#        [0.,0.66666667,0.33333333,0.]]


def cut_up_ballots(walk):
    result = []
    ballot = []
    for num in walk:
        if num != 0:
            ballot.append(num)
        elif ballot:
            result.append(ballot)
            ballot = []

    if ballot:  # In case the last element(s) are not followed by 0
        result.append(ballot)
    return result


def dedup(ballot):
    return list(OrderedDict.fromkeys(ballot))


def loop_erase(walk):
    result = []
    ballot = []
    for num in walk:
        if num != 0:
            if num in ballot:
                ballot = ballot[:ballot.index(num)] 
            ballot.append(num)
        elif ballot:
            result.append(ballot)
            ballot = []

    if ballot:  # In case the last element(s) are not followed by 0
        result.append(ballot)
        
    return result


if __name__ == '__main__':
    election, names, location = parse("Data/edinburgh17-16.blt")
    # mat = gen_transition_probs(election=election)
    # n = 1390
    # walk = sample_walk(mat, 1390)
    # ballots = cut_up_ballots(walk)
    # deduped = list(map(dedup, ballots))

# ballot = [2, 1, 2, 1, 3]
# print(dedup(ballot))

# print(deduped)
    
