from defcon import *
from zeidler1988 import ZeidlerProblem

def test_zeidler():
    problem = ZeidlerProblem()
    dc = DeflatedContinuation(problem, teamsize=1, verbose=True, clear_output=True)
    dc.run(values=dict(P=10.4, g=-1, a=1, rho=1))

    io = problem.io()
    io.setup(problem.parameters(), problem.functionals(), None)

    params = (10.4, 1, -1, 1)

    if backend.comm_world.rank == 0:
        assert len(io.known_branches(params)) == 3
