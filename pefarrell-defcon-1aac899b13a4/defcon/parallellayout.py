from __future__ import absolute_import, print_function

"""
Some utility functions that map teams to ranks and ranks to teams.

The default strategy is to group ranks sequentially, i.e. if teamsize = 4
and the ranks are

[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

then the teams will be

[10, 0, 0, 0, 0, 1, 1, 1, 1]

but this may not be optimal on your machine. If you want to implement a different
layout, just change these two functions.
"""

from mpi4py import MPI

import math


def ranktoteamno(rank, teamsize):
    if rank == 0:
        return MPI.COMM_WORLD.size
    return int(math.floor((rank-1)/float(teamsize)))

def teamnotoranks(teamno, teamsize):
    return range(teamno*teamsize+1, (teamno+1)*teamsize+1)

if __name__ == "__main__":
    teamsize = 4
    print("rank 0 -> team ", ranktoteamno(0, teamsize))
    print("rank 1 -> team ", ranktoteamno(1, teamsize))
    print("rank 2 -> team ", ranktoteamno(2, teamsize))
    print("rank 3 -> team ", ranktoteamno(3, teamsize))
    print("rank 4 -> team ", ranktoteamno(4, teamsize))
    print("rank 5 -> team ", ranktoteamno(5, teamsize))
    print("rank 6 -> team ", ranktoteamno(6, teamsize))
    print("rank 7 -> team ", ranktoteamno(7, teamsize))
    print("rank 8 -> team ", ranktoteamno(8, teamsize))
    print("team 0 -> ranks ", teamnotoranks(0, teamsize))
    print("team 1 -> ranks ", teamnotoranks(1, teamsize))
