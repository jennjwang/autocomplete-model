
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

