# -*- coding: utf-8 -*-
import sys
from   math import floor

from defcon import *
from dolfin import *

from petsc4py import PETSc

class LaplaceProblem(BifurcationProblem):
    def mesh(self, comm):
        return UnitSquareMesh(comm, 200, 200)

    def coarse_meshes(self, comm):
        return [UnitSquareMesh(comm, 25, 25), UnitSquareMesh(comm, 50, 50), UnitSquareMesh(comm, 100, 100)]

    def function_space(self, mesh):
        return FunctionSpace(mesh, "CG", 1)

    def parameters(self):
        f = Constant(0)

        return [(f, "f", r"$f$")]

    def residual(self, u, params, v):
        f = params[0]

        F = (
              inner(grad(u), grad(v))*dx
            - inner(f, v)*dx
            )

        return F

    def boundary_conditions(self, V, params):
        return [DirichletBC(V, 0.0, "on_boundary")]

    def functionals(self):
        def L2(u, params):
            j = sqrt(assemble(inner(u, u)*dx))
            return j

        return [(L2, "L2", r"$\|u\|$")]

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, V, params, n):
        return Function(V)

    def number_solutions(self, params):
        return 1

    def solver_parameters(self, params, task, **kwargs):
        args = {
               "snes_type": "ksponly",
               "snes_max_it": 10,
               "snes_atol": 1.0e-9,
               "snes_rtol": 1.0e-9,
               "snes_monitor": None,
               "snes_view": None,
               "ksp_type": "richardson",
               "ksp_monitor_true_residual": None,
               "ksp_atol": 1.0e-10,
               "ksp_rtol": 1.0e-10,
               "pc_type": "mg",
               "pc_mg_galerkin": None,
               }
        return args

if __name__ == "__main__":
    dc = DeflatedContinuation(problem=LaplaceProblem(), teamsize=1, verbose=True, clear_output=True, profile=True)
    dc.run(values={"f": [1.0]})
