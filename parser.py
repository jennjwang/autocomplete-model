
import pandas as pd
import numpy as np


def remove_zeros(ballot):
    to_return = []
    for vote in ballot:
        if vote != 0:
            to_return.append(vote)
    return tuple(to_return)


def parse(filename):
    election = {}
    names = []
    numbers = True
    with open(filename, "r") as file:
        for line in file:
            s = line.rstrip("\n").rstrip()
            if numbers:
                ballot = [int(vote) for vote in s.split(" ")]
                num_votes = ballot[0]
                if num_votes == 0:
                    numbers = False
                else:
                    election[remove_zeros(ballot[1:])] = num_votes
            elif "(" not in s:
                return election, names, s.strip("\"")
            else:
                name_parts = s.strip("\"").split(" ")
                first_name = " ".join(name_parts[:-2])
                last_name = name_parts[-2]
                party = name_parts[-1].strip("(").strip(")")
                names.append((first_name, last_name, party))
    raise Exception(f"Error parsing file '{filename}'.")


def parse_condorcet(filename):
    cvr = pd.read_csv(filename, index_col=0)
    cands = pd.unique(cvr[cvr.columns].values.ravel('K'))
    print(cands)
    cvr = cvr[cvr.columns].replace('skipped', np.nan)
    cvr = cvr[cvr.columns].replace('overvote', np.nan)
    cvr = cvr[cvr.columns].replace('writein', np.nan)
    cands = pd.unique(cvr[cvr.columns].values.ravel('K'))
    cands = cands[:-1]
    cands_dict = {name: idx + 1 for idx, name in enumerate(cands)}
    cvr = cvr[cvr.columns].replace(cands_dict)
    cvr = cvr.dropna(how='all')

    row_counts = cvr[cvr.columns].value_counts(dropna=False)
    df_counts = row_counts.reset_index()
    df_counts.columns = ['rank1', 'rank2', 'rank3', 'count']
    print(df_counts)


# # Save the DataFrame to a CSV file
#     df_counts.to_csv('minneapolis2021_ward2.csv', index=False)

# # processed = {tuple(value for value in key if not np.isnan(
# #     value)): row_counts[key] for key in row_counts}

# # return processed


# parse_condorcet("Data/condorcet/minneapolis_ward2_2021.csv")
