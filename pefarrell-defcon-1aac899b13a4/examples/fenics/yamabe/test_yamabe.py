from defcon import *
from yamabe import YamabeProblem

# Don't do the extra (failed) search
class TestYamabeProblem(YamabeProblem):
    def number_solutions(self, params):
        return 7

def test_yamabe():
    problem = TestYamabeProblem()
    deflation = ShiftedDeflation(problem, power=1, shift=1.0e-2)
    dc = DeflatedContinuation(problem, deflation=deflation, teamsize=1, verbose=True, clear_output=True)
    dc.run(values={"a": [8.0]})

    io = problem.io()
    io.setup(problem.parameters(), problem.functionals(), None)

    if backend.comm_world.rank == 0:
        assert len(io.known_branches((8.0,))) == 7
