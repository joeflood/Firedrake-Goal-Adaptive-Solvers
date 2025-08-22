# -*- coding: utf-8 -*-

# Normal form (10) from Golubitsky and Schaeffer, pg 196-209

from defcon import *
from dolfin import *

class GolubitskyProblem(BifurcationProblem):
    def mesh(self, comm):
        return UnitIntervalMesh(comm, 2)

    def function_space(self, mesh):
        return FunctionSpace(mesh, "R", 0)

    def parameters(self):
        lmbda = Constant(0)

        return [(lmbda, "lambda", r"$\lambda$")]

    def residual(self, x, params, v):
        lmbda = params[0]

        (alpha, beta, gamma) = (Constant(5.0), Constant(0.5), Constant(-5))

        F = (
              x**4*v*dx
            - lmbda*x*v*dx
            + alpha*v*dx
            + beta*lmbda*v*dx
            + gamma*x**2*v*dx
            )

        return F

    def boundary_conditions(self, V, params):
        return []

    def functionals(self):
        def value(x, params):
            return x.vector().array()[0]

        return [(value, "value", r"$x$")]

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, V, params, n):
        return interpolate(Constant(0.1), V)

    def number_solutions(self, params):
        lmbda = params[0]
        if lmbda < -1.3: return 2
        if lmbda > +0.7: return 2
        return 4

    def solver_parameters(self, params, task, **kwargs):
        args = {
               "snes_max_it": 50,
               "snes_atol": 1.0e-9,
               "snes_rtol": 0.0,
               "snes_monitor": None,
               "ksp_type": "preonly",
               "pc_type": "lu",
               }
        return args

if __name__ == "__main__":
    dc = DeflatedContinuation(problem=GolubitskyProblem(), teamsize=1, verbose=True, clear_output=True, logfiles=True)
    dc.run(values={"lambda": linspace(-10, 10, 2001)})
