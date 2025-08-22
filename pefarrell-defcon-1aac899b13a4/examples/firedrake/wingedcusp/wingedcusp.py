# -*- coding: utf-8 -*-
from firedrake import *
from defcon import *

class WingedCuspProblem(BifurcationProblem):
    def mesh(self, comm):
        return UnitIntervalMesh(1, comm=comm)

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
            with x.dat.vec_ro as r:
                return r.sum()

        return [(value, "value", r"$x$")]

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, V, params, n):
        return Function(V).interpolate(Constant(0.1))

    def number_solutions(self, params):
        lmbda = params[0]
        if 0.3 < lmbda < 4:
            return 3
        else:
            return 1

    def solver_parameters(self, params, task, **kwargs):
        args = {
               "mat_type": "matfree",
               "snes_max_it": 20,
               "snes_atol": 1.0e-9,
               "snes_rtol": 0.0,
               "snes_monitor": None,
               "snes_linesearch_type": "basic",
               "ksp_type": "gmres",
               "pc_type": "none",
               }
        return args

if __name__ == "__main__":
    dc = DeflatedContinuation(problem=WingedCuspProblem(), teamsize=1, verbose=True, clear_output=True)
    dc.run(values={"lambda": linspace(-3, 6, 901)})
