from defcon import *
from wingedcusp import WingedCuspProblem

def test_winged_cusp():
    problem = WingedCuspProblem()
    dc = DeflatedContinuation(problem, teamsize=1, clear_output=True)
    dc.run(values={"lambda": linspace(-3, 6, 91)})

    io = problem.io()
    io.setup(problem.parameters(), problem.functionals(), None)

    if backend.comm_world.rank == 0:
        assert len(io.known_branches((-2,))) == 1
        assert len(io.known_branches((+5,))) == 1
        assert len(io.known_branches((+1,))) == 3
        assert len(io.known_branches((+2,))) == 3
        assert len(io.known_branches((+3,))) == 3
