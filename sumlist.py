#!/usr/bin/env python3

# Paste your list here (can be nested), e.g.
# data = [1, 2, [3, 4], 5]
data = [12, 5, 8, 5, 8]


def sum_list(lst):
    total = 0
    for item in lst:
        if isinstance(item, list):
            total += sum_list(item)
        else:
            total += item
    return total


print(sum_list(data))
