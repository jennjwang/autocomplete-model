from parser import parse
from clustering import *
from tqdm import tqdm
from intervals import *
import numpy as np 
import matplotlib.pyplot as plt


has_plot = False


def visualize_intervals_borda(filename, bg_cutoff, num_repetitions, radius, discount):
    global has_plot
    has_plot = True
    election, names, location = parse(filename)
    n = len(names)
    for i in range(n):
        print(f"{i + 1}: {names[i]}")
    bg = ballot_graph(n, bg_cutoff, election)
    centers_dict = k_means(bg, n, bg_cutoff, 2, num_repetitions=num_repetitions)
    a = sorted([(-occurrences, centers) for centers, occurrences in centers_dict.items()])
    num_plots = len(a)
    extra = num_plots == 1
    if extra:
        num_plots = 2
    fig, axs = plt.subplots(num_plots)
    for i in range(num_plots):
        occurrences, centers = a[i]
        intervals = [get_interval_borda( \
                    n, bg, center, radius=radius, discount=discount) \
                    for center in centers]
        X_axis = np.arange(n + 1)
        if intervals[0][1] >= intervals[1][1]:
            i1 = 0
            i2 = 1
        else:
            i1 = 1
            i2 = 0
        axs[i].bar(X_axis - 0.2, intervals[i1], 0.4)
        axs[i].bar(X_axis + 0.2, intervals[i2], 0.4)
        axs[i].text(0, max(intervals[0] + intervals[1])/2, str(-occurrences), ha="right")
        if extra:
            return


def visualize_intervals_iac(filename, bg_cutoff, num_repetitions, discount):
    global has_plot
    has_plot = True
    election, names, location = parse(filename)
    n = len(names)
    for i in range(n):
        print(f"{i + 1}: {names[i]}")
    bg = ballot_graph(n, bg_cutoff, election)
    intervals_dict = interval_aware_clustering(bg, n, bg_cutoff, 2, discount, num_repetitions=num_repetitions)
    a = sorted([(-occurrences, tuple(intervals)) for intervals, occurrences in intervals_dict.items()])
    num_plots = len(a)
    extra = num_plots == 1
    if extra:
        num_plots = 2
    fig, axs = plt.subplots(num_plots)
    for i in range(num_plots):
        occurrences, intervals = a[i]
        X_axis = np.arange(n + 1)
        if intervals[0][1] >= intervals[1][1]:
            i1 = 0
            i2 = 1
        else:
            i1 = 1
            i2 = 0
        axs[i].bar(X_axis - 0.2, intervals[i1], 0.4)
        axs[i].bar(X_axis + 0.2, intervals[i2], 0.4)
        axs[i].text(0, max(intervals[0] + intervals[1])/2, str(-occurrences), ha="right")
        if extra:
            return

