from defcon import *
from obstacle import ObstacleProblem

def test_obstacle():
    problem = ObstacleProblem()
    dc = DeflatedContinuation(problem, teamsize=1, verbose=True, clear_output=True)
    dc.run(values={"f": -10.0, "scale": 1.0})

    io = problem.io()
    io.setup(problem.parameters(), problem.functionals(), None)

    params = (-10.0, 1.0)

    if backend.comm_world.rank == 0:
        assert len(io.known_branches(params)) == 1

    functionals = io.fetch_functionals([params], 0)[0]
    assert 0.65 < functionals[0] < 0.66
