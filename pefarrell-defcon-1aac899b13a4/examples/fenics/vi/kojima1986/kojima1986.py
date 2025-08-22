"""
Nonlinear complementarity problem with two solutions.

Drawn from

@article{kojima1986,
title = "Extensions of {N}ewton and quasi-{N}ewton methods to systems of {PC}$^1$ equations",
journal = "Journal of Operations Research Society of Japan",
volume = "29",
pages = "352--374",
year = "1986",
author = "M. Kojima and S. Shindo",
}

"""

from dolfin import *
from defcon import *

N = 4
def F(z, params):
    z1 = z[0]
    z2 = z[1]
    z3 = z[2]
    z4 = z[3]

    F1 = 3*z1**2 + 2*z1*z2 + 2*z2**2 + z3 + 3*z4 - 6
    F2 = 2*z1**2 + z2**2  + z1 + 10*z3 + 2*z4 - 2
    F3 = 3*z1**2 + z1*z2 + 2*z2**2 + 2*z3 + 9*z4 - 9
    F4 =   z1**2 + 3*z2**2 + 2*z3 + 3*z4 - 3
    F = [F1, F2, F3, F4]

    return F

class KojimaProblem(ComplementarityProblem):
    def parameters(self):
        lamda = Constant(0)
        return [(lamda, "lambda", r"$\lambda$")]

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, V, params, n):
        c = 0.7
        return interpolate(Constant((c, c, c, c)), V)

    def number_solutions(self, params):
        return 2

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
    problem = KojimaProblem(F, N)
    deflation = ShiftedDeflation(problem, power=2, shift=1)
    dc = DeflatedContinuation(problem, deflation=deflation, teamsize=1, clear_output=True)
    dc.run(values={"lambda": 0})
