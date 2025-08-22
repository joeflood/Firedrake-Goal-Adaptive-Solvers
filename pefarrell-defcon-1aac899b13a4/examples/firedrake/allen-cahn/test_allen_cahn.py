from defcon import *
ac = __import__("allen-cahn")

def test_allen_cahn():
    problem = ac.AllenCahnProblem()
    dc = DeflatedContinuation(problem, teamsize=1, verbose=True, clear_output=True)
    dc.run(values={"delta": [0.04]})

    io = problem.io()
    io.setup(problem.parameters(), problem.functionals(), None)

    if backend.comm_world.rank == 0:
        assert len(io.known_branches((0.04,))) == 3
