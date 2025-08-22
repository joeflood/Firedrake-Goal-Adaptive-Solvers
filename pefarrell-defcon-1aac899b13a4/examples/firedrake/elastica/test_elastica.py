from defcon import *
from elastica import ElasticaProblem
from math import pi
import matplotlib.pyplot as plt
from mpi4py import MPI


def test_elastica():
    problem = ElasticaProblem()
    dc = DeflatedContinuation(problem, teamsize=1, clear_output=True)
    values = linspace(0, 1.1*pi, 10)
    dc.run(values={"lambda": values, "mu": [0.5]}, freeparam="lambda")

    io = problem.io(comm=MPI.COMM_SELF)

    V = problem.function_space(problem.mesh(MPI.COMM_SELF))

    io.setup(problem.parameters(), problem.functionals(), V)

    if backend.comm_world.rank == 0:
        final = (values[-1], 0.5)
        branches = io.known_branches(final)
        len_branches = len(branches)

        stabilities = io.fetch_stability(final, branches)
        print("stabilities: %s" % stabilities)
        # Two stable, one unstable
        sum_stab = sum(stab["stable"] for stab in stabilities)
    else:
        len_branches = 0
        sum_stab = 0

    # We hang if we were to just assert on rank 0 and the assertion fails
    assert backend.comm_world.bcast(len_branches) == 3
    assert backend.comm_world.bcast(sum_stab) == 2

    # Check that this doesn't crash
    dc.bifurcation_diagram("signedL2", fixed={"mu": 0.5})
    plt.title(r"Buckling of an Euler elastica, $\mu = 1/2$")
    plt.savefig("bifurcation.pdf")
