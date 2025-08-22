from __future__ import print_function

from dolfin import *
from defcon import *

import numpy as np
import collections
from matplotlib import pyplot as plt

N = 42
b = np.loadtxt(open("b.txt", "r"))
A = collections.defaultdict(list)
data = np.loadtxt(open("A.txt", "r"))
for (i, j, v) in data:
    i = int(i - 1)
    j = int(j - 1)
    coeffs = A[i]
    coeffs.append((j, v))
    A[i] = coeffs

def F(z, params):
    q = params[0]

    out = []
    for i in range(N):
        form = b[i]
        for (j, v) in A[i]:
            form += q*v*z[j]
        out.append(form)

    return out

class TinLoiProblem(ComplementarityProblem):
    def parameters(self):
        q = Constant(0)
        return [(q, "q", r"$q$")]

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, V, params, n):
        guess = Function(V)
        guess.vector()[:] = 0.4
        return guess

    def number_solutions(self, params):
        return 2

    def solver_parameters(self, params, task, **kwargs):
        args = {
               "snes_max_it": 500,
               "snes_atol": 1.0e-9,
               "snes_rtol": 0.0,
               "snes_stol": 0.0,
               "snes_converged_reason": None,
               "snes_monitor": None,
               "snes_linesearch_damping": 1.0,
               "snes_linesearch_maxstep": 1.0,
               "snes_linesearch_type": "l2",
               "snes_linesearch_monitor": None,
               "ksp_type": "preonly",
               "pc_type": "svd",
               }
        return args

if __name__ == "__main__":
    problem = TinLoiProblem(F, N)
    deflation = ShiftedDeflation(problem, power=1, shift=1)
    dc = DeflatedContinuation(problem, deflation=deflation, teamsize=1, clear_output=True)
    dc.run(values={"q": [1.0]})
