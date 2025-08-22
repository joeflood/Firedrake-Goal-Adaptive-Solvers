"""
Linear multiplicative programming problem.

Drawn from

@article{konno1992,
year={1992},
journal={Mathematical Programming},
volume={56},
number={1--3},
doi={10.1007/BF01580893},
title={Linear multiplicative programming},
author={Konno, H. and Kuno, T.},
pages={51--64},
}
"""

from dolfin import *
from defcon import *

N = 9

A = as_matrix([[-0.2, -0.4],
               [0.28, -0.28],
               [0.35, 0.35],
               [0.56, 0.28],
               [0.583333333333333, 0.0],
               [-0.430769230769231, 0.107692307692308],
               [-0.451612903225806, -0.225806451612903]])
v = as_vector([1.2, 0.84, 0.7, 0.56, 0.583333333333333, 1.292307692307692, 1.354838709677419])

def F(z, params):

    l = as_vector([z[i] for i in range(2, 9)]) # Lagrange multipliers
    x = as_vector([z[i] for i in [0, 1]]) # State
    f = [0]*N

    ATl = dot(A.T, l)
    Ax  = dot(A,   x)

    f[0] = +2*x[0] + ATl[0]
    f[1] = -2*x[0] + ATl[1]
    f[2:9] = v - Ax

    return f

class KonnoProblem(ComplementarityProblem):
    def parameters(self):
        lamda = Constant(0)
        return [(lamda, "lambda", r"$\lambda$")]

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, V, params, n):
        c = 2
        init = Constant(as_vector([c]*N))
        return interpolate(init, V)

    def number_solutions(self, params):
        return 3

    def solver_parameters(self, params, task, **kwargs):
        args = {
               "snes_max_it": 100,
               "snes_atol": 1.0e-9,
               "snes_rtol": 0.0,
               "snes_stol": 0.0,
               "snes_converged_reason": None,
               "snes_monitor": None,
               "snes_linesearch_damping": 0.9,
               "snes_linesearch_maxstep": 1.0,
               "snes_linesearch_type": "l2",
               "ksp_type": "preonly",
               "pc_type": "svd",
               }
        return args

if __name__ == "__main__":
    problem = KonnoProblem(F, N)
    dc = DeflatedContinuation(problem, teamsize=1, clear_output=True)
    dc.run(values={"lambda": 0})
