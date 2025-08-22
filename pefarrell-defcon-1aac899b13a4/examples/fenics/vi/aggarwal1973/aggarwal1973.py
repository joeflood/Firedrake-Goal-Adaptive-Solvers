"""
Bimatrix game with three equilibria.

Drawn from

@article{aggarwal1973,
year={1973},
journal={Mathematical Programming},
volume={4},
number={1},
doi={10.1007/BF01584663},
title={On the generation of all equilibrium points for bimatrix games through the {Lemke--Howson} Algorithm},
author={Aggarwal, V.},
pages={233--234},
}
"""

from __future__ import print_function

from dolfin import *
from defcon import *

from matplotlib import pyplot as plt

N = 4
def F(z, params):
    q = params[0]
    x1 = z[0]
    x2 = z[1]
    y1 = z[2]
    y2 = z[3]

    F1 = q*30*y1 + q*20*y2 - 1
    F2 = q*10*y1 + q*25*y2 - 1
    F3 = q*30*x1 + q*20*x2 - 1
    F4 = q*10*x1 + q*25*x2 - 1
    F = [F1, F2, F3, F4]

    return F

class AggarwalProblem(ComplementarityProblem):
    def parameters(self):
        lamda = Constant(0)
        return [(lamda, "lambda", r"$\lambda$")]

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, V, params, n):
        return interpolate(Constant([0, 0, 0.0, 0]), V)

    def number_solutions(self, params):
        if params[0] == 0.0: return 1
        else: return 3

    def solver_parameters(self, params, task, **kwargs):
        args = {
               "snes_max_it": 100,
               "snes_atol": 1.0e-9,
               "snes_rtol": 0.0,
               "snes_stol": 0.0,
               "snes_converged_reason": None,
               "snes_monitor": None,
               "snes_linesearch_damping": 1.0,
               "snes_linesearch_maxstep": 1.0,
               "snes_linesearch_type": "basic",
               "ksp_type": "preonly",
               "pc_type": "svd",
               }
        return args

if __name__ == "__main__":
    problem = AggarwalProblem(F, N)
    deflation = ShiftedDeflation(problem, power=2, shift=1)
    dc = DeflatedContinuation(problem, deflation=deflation, teamsize=1, clear_output=True)
    start = 0.001
    lambdas = [start] + list(linspace(start, 1, 51)[1:])
    dc.run(values={"lambda": lambdas})

    dc.bifurcation_diagram("l2norm")
    plt.title(r"The Aggarwal bimatrix game")
    plt.savefig("bifurcation.pdf")

    if backend.comm_world.rank == 0:
        print()
        params = (1,)
        branches = dc.thread.io.known_branches(params)
        for branch in branches:
            functionals = dc.thread.io.fetch_functionals([params], branch)[0]
            print("Solution: %s" % functionals)
