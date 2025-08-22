from defcon import *
from konno1992 import KonnoProblem, N, F

def test_konno():
    problem = KonnoProblem(F, N)
    dc = DeflatedContinuation(problem, teamsize=1, verbose=True, clear_output=True)
    dc.run(values={"lambda": 0})

    io = problem.io()
    io.setup(problem.parameters(), problem.functionals(), None)

    if backend.comm_world.rank == 0:
        params = (0,)
        assert len(io.known_branches(params)) == 3
