from defcon import *
from gerard import GerardProblem
import pytest
import petsc4py

@pytest.mark.skip(reason="Doesn't work with old versions of petsc4py used in FEniCS docker images")
def test_gerard():
    problem = GerardProblem()
    deflation = ShiftedDeflation(problem, power=1, shift=1)
    dc = DeflatedContinuation(problem, deflation=deflation, teamsize=1, verbose=True, clear_output=True)
    dc.run(values={"f": 0})

    io = problem.io()
    io.setup(problem.parameters(), problem.functionals(), None)

    if backend.comm_world.rank == 0:
        params = (0,)
        assert len(io.known_branches(params)) == 3
