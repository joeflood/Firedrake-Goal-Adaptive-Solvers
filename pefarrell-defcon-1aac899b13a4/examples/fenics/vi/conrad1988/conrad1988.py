# -*- coding: utf-8 -*-

"""
Example from


@article{conrad1988,
author = {F. Conrad and R. Herbin and H. D. Mittelmann},
title = {Approximation of Obstacle Problems by Continuation Methods},
journal = {SIAM Journal on Numerical Analysis},
volume = {25},
number = {6},
pages = {1409--1431},
year = {1988},
doi = {10.1137/0725082},
}
"""

from defcon import *
from dolfin import *

class BratuProblem(BifurcationProblem):
    def mesh(self, comm):
        mesh = RectangleMesh(comm, Point(-1, -1), Point(1, 1), 100, 100, "crossed")

        return mesh

    def function_space(self, mesh):
        return FunctionSpace(mesh, "CG", 1)

    def parameters(self):
        lamda = Constant(0)
        return [(lamda, "lambda", r"$\lambda$")]

    def residual(self, u, params, v):
        lamda = params[0]

        F = (
              inner(grad(u), grad(v))*dx
            - lamda*exp(u)*v*dx
            )

        return F

    def boundary_conditions(self, V, params):
        return [DirichletBC(V, 0, "on_boundary")]

    def functionals(self):
        def L2norm(u, params):
            j = sqrt(assemble(inner(u, u)*dx))
            return j

        def H1norm(u, params):
            j = sqrt(assemble(inner(grad(u), grad(u))*dx + inner(u, u)*dx))
            return j

        return [(L2norm, "L2norm", r"$\|u\|_{L^2}$"),
                (H1norm, "H1norm", r"$\|u\|_{H^1}$")]

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, V, params, n):
        return Function(V)

    def number_solutions(self, params):
        if params[0] == 1.5:
            return 1
        elif params[0] > 2.04:
            return 1
        else:
            return 3

    def squared_norm(self, a, b, params):
        return inner(a - b, a - b)*dx + inner(grad(a - b), grad(a - b))*dx

    def solver_parameters(self, params, task, **kwargs):
        args = {
               "snes_max_it": 100,
               "snes_atol": 1.0e-8,
               "snes_rtol": 0.0,
               "snes_stol": 0.0,
               "snes_converged_reason": None,
               "snes_divergence_tolerance": 1.0e10,
               "snes_linesearch_type": "basic",
               "snes_linesearch_maxstep": 1.0,
               "snes_linesearch_damping": 1.00,
               "snes_linesearch_monitor": None,
               "snes_monitor": None,
               "ksp_type": "preonly",
               "pc_type": "lu",
               "pc_factor_mat_solver_package": "mumps",
               "pc_factor_mat_solver_type": "mumps",
               "mat_mumps_icntl_24": 1,
               "mat_mumps_icntl_13": 1
               }
        return args

    def bounds(self, V, params, initial_guess):
        lb = Constant(-1e20)
        ub = Expression("1 + 5*sqrt(x[0]*x[0] + x[1]*x[1])", degree=1, mpi_comm=V.mesh().mpi_comm())

        l = interpolate(lb, V)
        u = interpolate(ub, V)
        return (l, u)

if __name__ == "__main__":
    problem = BratuProblem()
    dc = DeflatedContinuation(problem=problem, teamsize=1, verbose=True, clear_output=True, logfiles=False)

    # Continue in lambda. The folds lie between [2.038, 2.04] and [1.575, 1.58]
    lambdas = list(arange(1.5, 1.575, 0.00125)) + list(linspace(1.575, 1.58, 20))[:-1] + list(arange(1.58, 2.038, 0.00125)) + list(linspace(2.038, 2.04, 20))[:-1] + list(arange(2.04, 3.0, 0.00125))
    dc.run(values={"lambda": lambdas}, freeparam="lambda")
