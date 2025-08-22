# -*- coding: utf-8 -*-
from defcon import *
from dolfin import *

class WingedCuspProblem(BifurcationProblem):
    def mesh(self, comm):
        return UnitIntervalMesh(comm, 2)

    def function_space(self, mesh):
        return FunctionSpace(mesh, "R", 0)

    def parameters(self):
        lmbda = Constant(0)

        return [(lmbda, "lambda", r"$\lambda$")]

    def residual(self, x, params, v):
        lmbda = params[0]

        F = (
              x**3*v*dx
            - 2*lmbda*x*v*dx
            + lmbda**2*v*dx
            - 2*lmbda*v*dx
            + 1*v*dx
            )

        return F

    def boundary_conditions(self, V, params):
        return []

    def functionals(self):
        def value(x, params):
            return x.vector().get_local()[0]

        return [(value, "value", r"$x$")]

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, V, params, n):
        return interpolate(Constant(0.1), V)

    def number_solutions(self, params):
        lmbda = params[0]
        if 0.3 < lmbda < 4:
            return 3
        else:
            return 1

    def solver_parameters(self, params, task, **kwargs):
        args = {
               "snes_max_it": 20,
               "snes_atol": 1.0e-9,
               "snes_rtol": 0.0,
               "snes_monitor": None,
               "ksp_type": "preonly",
               "pc_type": "lu",
               }
        return args

if __name__ == "__main__":
    dc = DeflatedContinuation(problem=WingedCuspProblem(), teamsize=1, verbose=True, clear_output=True)
    dc.run(values={"lambda": linspace(-3, 6, 901)})
    #dc.run(values={"lambda": [0.54]})
