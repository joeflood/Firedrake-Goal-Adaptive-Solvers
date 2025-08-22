from defcon import *
from hyperelasticity import HyperelasticityProblem
from mpi4py import MPI


def test_elastica():
    problem = HyperelasticityProblem()
    dc = DeflatedContinuation(problem, teamsize=1, clear_output=True)
    values = list(arange(0.0, 0.05, 0.005)) + [0.05]
    dc.run(values={"eps": values})

    io = problem.io(comm=MPI.COMM_SELF)
    V = problem.function_space(problem.mesh(MPI.COMM_SELF))
    io.setup(problem.parameters(), problem.functionals(), V)

    if backend.comm_world.rank == 0:
        final = (values[-1], 0.05)
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
