# -*- coding: utf-8 -*-
import sys
from   math import floor

from defcon import *
from dolfin import *

class AllenCahnProblem(BifurcationProblem):
    def mesh(self, comm):
        return UnitSquareMesh(comm, 100, 100, "crossed")

    def coarse_meshes(self, comm):
        # For maximal efficiency, we would use nested meshes (i.e. "crossed").
        # But we want to test here the GMG efficiency with non-nested meshes.
        return [UnitSquareMesh(comm, 25, 25, "left"), UnitSquareMesh(comm, 50, 50, "right")]

    def function_space(self, mesh):
        V = FunctionSpace(mesh, "CG", 1)

        # Construct BCs here for efficiency
        self._bcs = [DirichletBC(V, +1.0, "x[0] == 0.0 || x[0] == 1"),
                     DirichletBC(V, -1.0, "x[1] == 0.0 || x[1] == 1")]
        return V

    def parameters(self):
        delta = Constant(0)
        return [(delta, "delta", r"$\delta$")]

    def residual(self, y, params, v):
        delta = params[0]

        F = (
            + delta * inner(grad(v), grad(y))*dx
            + 1.0/delta * inner(v, y**3 - y)*dx
            )

        return F

    def boundary_conditions(self, V, params):
        return self._bcs

    def functionals(self):
        def sqL2(y, params):
            j = assemble(y*y*dx)
            return j

        return [(sqL2, "sqL2", r"$\|y\|^2$")]

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, V, params, n):
        return Function(V)

    def number_solutions(self, params):
        delta = params[0]
        if delta == 0.04: return 3
        else:             return float("inf")

    def solver_parameters(self, params, task, **kwargs):
        return {
               "snes_max_it": 50,
               "snes_atol": 1.0e-9,
               "snes_rtol": 1.0e-9,
               "snes_monitor": None,
               "ksp_type": "gmres",
               "ksp_monitor": None,
               "ksp_rtol": 1.0e-10,
               "ksp_atol": 1.0e-10,
               "pc_type": "mg",
               "pc_mg_galerkin": None,
               "mg_levels_ksp_type": "chebyshev",
               "mg_levels_ksp_max_it": 2,
               "mg_levels_pc_type": "sor",
               }

if __name__ == "__main__":
    dc = DeflatedContinuation(problem=AllenCahnProblem(), teamsize=1, verbose=True, clear_output=True)
    dc.run(values={"delta": [0.04]}, freeparam="delta")
