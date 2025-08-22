# -*- coding: utf-8 -*-
import sys

from defcon import *
from dolfin import *

parameters["form_compiler"]["quadrature_degree"] = 5
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -ffast-math -march=native"

class YamabeProblem(BifurcationProblem):
    def mesh(self, comm):
        return Mesh(comm, "mesh/doughnut.xml")

    def coarse_meshes(self, comm):
        return [Mesh(comm, "mesh/doughnut.2.xml"), Mesh(comm, "mesh/doughnut.1.xml")]

    def function_space(self, mesh):
        return FunctionSpace(mesh, "CG", 1)

    def parameters(self):
        a = Constant(0)
        return [(a, "a", r"$a$")]

    def residual(self, y, params, v):
        a = params[0]

        x = SpatialCoordinate(y.function_space().mesh())
        r = sqrt(x[0]**2 + x[1]**2)
        rho = 1.0/r**3

        F = (
             a*inner(grad(y), grad(v))*dx +
             rho * inner(y**5, v)*dx +
             Constant(-1.0/10.0)*inner(y, v)*dx
            )

        return F

    def boundary_conditions(self, V, params):
        bcs = [DirichletBC(V, Constant(1.0), "on_boundary")]
        return bcs

    def functionals(self):
        def sqL2(y, params):
            j = assemble(y*y*dx)
            return j

        return [(sqL2, "sqL2", r"$\|y\|^2$")]

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, V, params, n):
        return interpolate(Constant(1), V)

    def solver_parameters(self, params, task, **kwargs):
        return {
               "snes_max_it": 100,
               "snes_stol": 0.0,
               "snes_rtol": 1.0e-10,
               "snes_trtol": 0.0,
               "snes_monitor": None,
               "snes_divergence_tolerance": -1,
               "ksp_monitor": None,
               "ksp_type": "gcr",
               "ksp_gmres_restart": 100,
               "ksp_max_it": 1000,
               "ksp_atol": 1.0e-10,
               "ksp_rtol": 1.0e-10,
               "pc_type": "mg",
               "pc_mg_galerkin": None,
               "pc_gamg_verbose": 10,
               "pc_gamg_type": "agg",
               "pc_gamg_coarse_eq_limit": 2000,
               "pc_gamg_agg_nsmooths": 4,
               "pc_gamg_threshold": 0.04,
               "pc_gamg_square_graph": 1,
               "pc_gamg_sym_graph": 1,
               "mg_coarse_pc_type": "redundant",
               "mg_coarse_sub_pc_type": "lu",
               "mg_levels_pc_type": "jacobi",
               "mg_levels_ksp_type": "chebyshev",
               "mg_levels_ksp_max_it": 10,
               "gamg_est_ksp_max_it": 30,
               }

if __name__ == "__main__":
    problem = YamabeProblem()
    deflation = ShiftedDeflation(problem, power=1, shift=1.0e-2)
    dc = DeflatedContinuation(problem, deflation=deflation, teamsize=1, verbose=True, clear_output=True)
    dc.run(values={"a": [8.0]})
