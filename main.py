from parser import parse
from clustering import *
from tqdm import tqdm
from intervals import *
import numpy as np 
import matplotlib.pyplot as plt
import plotting
from plotting import *


test_num = 20

if test_num == 1:
    election, names, location = parse("Data/edinburgh17-16.blt")
    print(len(election))
    print(election[(1, 2)])
    print(names)
    print(location)
elif test_num == 2:
    bg = ballot_graph(3, 3)
    draw(bg)
elif test_num == 3:
    election = {(1, 2, 3, 4, 5, 6): 20, (1, 2, 3): 30, (1, 2): 7}
    bg = ballot_graph(5, 3, election)
    for n in bg.nodes(data=True):
        print(n)
elif test_num == 4:
    election, names, location = parse("Data/edinburgh17-16.blt")
    n = len(names)
    bg = ballot_graph(n, 3, election)
    for n in bg.nodes(data=True):
        print(n)
elif test_num == 5:
    election, names, location = parse("Data/edinburgh17-16.blt")
    n = len(names)
    for i in range(n):
        print(f"{i + 1}: {names[i]}")
    bg = ballot_graph(n, 2, election)
    print("\nk-means centers:")
    print(k_means(bg, n, 2, 2))
elif test_num == 6:
    election, names, location = parse("Data/edinburgh17-16.blt")
    n = len(names)
    bg = ballot_graph(4, 2)
    draw(bg)
elif test_num == 7:
    election, names, location = parse("Data/edinburgh17-16.blt")
    n = len(names)
    bg = ballot_graph(n, 2, election)
    draw(bg)
elif test_num == 8:
    bg = ballot_graph(10, 10)
    print("Got graph")
elif test_num == 9:
    for n in tqdm(range(4, 10)):
        pickle_distance_matrix(n, 4)
elif test_num == 10:
    election, names, location = parse("Data/edinburgh17-16.blt")
    n = len(names)
    for i in range(n):
        print(f"{i + 1}: {names[i]}")
    bg = ballot_graph(n, 4, election)
    print("\nk-means centers:")
    print(k_means(bg, n, 4, 2))
elif test_num == 11:
    for election_num in range(1, 18):
        print(f"\n\nElection {election_num}:\n")
        election, names, location = parse(f"Data/edinburgh17-{election_num:02}.blt")
        n = len(names)
        for i in range(n):
            print(f"{i + 1}: {names[i]}")
        bg = ballot_graph(n, 4, election)
        print("\nk-means centers:")
        try:
            print(k_means(bg, n, 4, 2))
        except:
            print("Can't do it, too many candidates.")
elif test_num == 12:
    election, names, location = parse("Data/edinburgh17-16.blt")
    n = len(names)
    for i in range(n):
        print(f"{i + 1}: {names[i]}")
    bg = ballot_graph(n, 4, election)
    centers = k_means(bg, n, 4, 2)
    intervals = {center: get_interval_borda( \
                n, bg, center, radius=2, discount=1) \
                for center in centers}
    print(intervals)
elif test_num == 13:
    visualize_intervals_borda("Data/edinburgh17-16.blt", bg_cutoff=4,
                              num_repetitions=20, radius=2, discount=1)
elif test_num == 14:
    visualize_intervals_borda("Data/edinburgh17-16.blt", bg_cutoff=4,
                              num_repetitions=20, radius=2, discount=.5)
elif test_num == 15:
    visualize_intervals_borda("Data/edinburgh17-16.blt", bg_cutoff=4,
                              num_repetitions=20, radius=3, discount=1)
elif test_num == 16:  # Plot intervals from k-means cluster + borda for each election.
    for election_num in range(1, 18):
        print(f"\n\nElection {election_num}:\n")
        try:
            visualize_intervals_borda(f"Data/edinburgh17-{election_num:02}.blt", bg_cutoff=4,
                                      num_repetitions=50, radius=2, discount=.75)
            plt.savefig(f"Plots/T{test_num}-{election_num:02}.png")
        except:
            pass
    plotting.has_plot = False
elif test_num == 17:
    for n in tqdm(range(5, 8)):
        pickle_distance_matrix(n, 5)
elif test_num == 18:  # Same as test 16 but truncating to 5 instead of 4. CORRUPTED PLOTS
    for election_num in range(1, 18):
        print(f"\n\nElection {election_num}:\n")
        try:
            visualize_intervals_borda(f"Data/edinburgh17-{election_num:02}.blt", bg_cutoff=5,
                                      num_repetitions=50, radius=2, discount=.75)
            plt.savefig(f"Plots/T{test_num}-{election_num:02}.png")
        except:
            pass
    plotting.has_plot = False
elif test_num == 19:  # Same as test 16 but with interval-aware clustering algorithm.
    for election_num in range(1, 18):
        print(f"\n\nElection {election_num}:\n")
        try:
            visualize_intervals_iac(f"Data/edinburgh17-{election_num:02}.blt", bg_cutoff=4,
                                      num_repetitions=50, discount=.75)
            plt.savefig(f"Plots/T{test_num}-{election_num:02}.png")
        except FileNotFoundError as e:
            pass
    plotting.has_plot = False
elif test_num == 20:  # Made during July 26 meeting, like test 16 but not cutting off 2-nbhds.
    for election_num in range(1, 18):
        print(f"\n\nElection {election_num}:\n")
        try:
            visualize_intervals_borda(f"Data/edinburgh17-{election_num:02}.blt", bg_cutoff=4,
                                      num_repetitions=50, radius=2000, discount=.75)
            plt.savefig(f"Plots/T{test_num}-{election_num:02}.png")
        except:
            pass
    plotting.has_plot = False


if plotting.has_plot:
    plt.savefig(f"Plots/T{test_num}.png")
