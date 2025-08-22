from defcon import *
from kojima1986 import KojimaProblem, N, F

def test_gould():
    problem = KojimaProblem(F, N)
    dc = DeflatedContinuation(problem, teamsize=1, verbose=True, clear_output=True)
    dc.run(values={"lambda": 0})

    io = problem.io()
    io.setup(problem.parameters(), problem.functionals(), None)

    if backend.comm_world.rank == 0:
        params = (0,)
        assert len(io.known_branches(params)) == 2
