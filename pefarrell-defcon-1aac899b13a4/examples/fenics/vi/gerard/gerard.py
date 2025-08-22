# -*- coding: utf-8 -*-
"""
Economic problem with three equilibria, taken from https://arxiv.org/abs/1706.08398
and a GAMS formulation supplied by Prof. Michael Ferris
"""

from __future__ import print_function
from defcon import *
from dolfin import *
import numpy

inf = 1e20
N = 10

def packer(vec):
    class DolfinIsClumsySometimes(UserExpression):
        def eval(self, values, x):
            values[:] = vec
        def value_shape(self):
            return (N,)
    return DolfinIsClumsySometimes(degree=0)

lb = packer([0, 0, 0, 0, 0, 0, 0, -inf, 0, 0])
ub = packer([inf]*N)

class GerardProblem(BifurcationProblem):
    def mesh(self, comm):
        return UnitIntervalMesh(comm, 1)

    def function_space(self, mesh):
        V = VectorFunctionSpace(mesh, "R", 0, dim=N)
        return V

    def parameters(self):
        f = Constant(0)
        return [(f, "f", r"$f$")]

    def equations(self, u):
        (x0, x11, x12, y1, y2, pi1, pi2, thetaP, u4, u5) = u
        sqr = lambda z: z**2

        eqns = [
          (-(0.75*(pi1 - 11.5*x0) + 0.25*(pi2 - 11.5*x0)))*u4 + (-(0.25*(pi1 - 11.5*x0) + 0.75*(pi2 - 11.5*x0)))*u5,
          (-0.75*(pi1 - x11))*u4 + (-0.25*(pi1 - x11))*u5,
          (-0.25*(pi2 - 3.5*x12))*u4 + (-0.75*(pi2 - 3.5*x12))*u5,
          (-(4 - pi1 - 2*y1))/(1),
          (-(9.6 - pi2 - 10*y2))/(1),
          x0 + x11 - y1,
          x0 + x12 - y2,
          -1 + u4 + u5,   # enforced as an equation
          (0.75*(pi1*(x0 + x11) - 5.75*sqr(x0) - 0.5*sqr(x11)) + 0.25*(pi2*(x0 + x12) - 5.75*sqr(x0) - 1.75*sqr(x12))) - thetaP,
          (0.25*(pi1*(x0 + x11) - 5.75*sqr(x0) - 0.5*sqr(x11)) + 0.75*(pi2*(x0 + x12) - 5.75*sqr(x0) - 1.75*sqr(x12))) - thetaP
               ]

        return eqns

    def residual(self, u, params, v):
        eqns = self.equations(split(u))
        F = 0
        for (e, w) in zip(eqns, split(v)):
            F += inner(e, w)*dx

        return F

    def boundary_conditions(self, V, params):
        return []

    def functionals(self):
        def pi1(u, params):
            return u((0.5,))[5]

        def pi2(u, params):
            return u((0.5,))[6]

        return [(pi1, "pi1", r"$\pi_1$"),
                (pi2, "pi2", r"$\pi_2$")]

    def number_initial_guesses(self, params):
        return 1

    def number_solutions(self, params):
        return 3

    def initial_guess(self, V, params, n):
        guess = [0]*N
        return interpolate(packer(guess), V)

    def monitor(self, params, branchid, solution, functionals):
        u = solution.vector().get_local()
        r = self.equations(u)
        print("Iterate:  %s" % list(u))
        print("Residual: %s" % self.equations(u))
        print("Product:  %s" % list(numpy.array(r) * u))

    def solver_parameters(self, params, task, **kwargs):
        return {
               "snes_max_it": 100,
               "snes_max_funcs": 1000000,
               "snes_linesearch_type": "l2",
               "snes_linesearch_damping": 0.9,
               "snes_linesearch_maxstep": 1.0,
               "snes_linesearch_monitor": None,
               "snes_atol": 1.0e-9,
               "snes_rtol": 0.0,
               "snes_monitor": None,
               "ksp_type": "preonly",
               "ksp_monitor": None,
               "ksp_rtol": 1.0e-10,
               "ksp_atol": 1.0e-10,
               "pc_type": "svd",
               "pc_factor_mat_solver_package": "mumps",
               "pc_factor_mat_solver_type": "mumps",
               }

    def bounds(self, V, params, initial_guess):
        l = interpolate(lb, V)
        u = interpolate(ub, V)
        return (l, u)

if __name__ == "__main__":
    problem = GerardProblem()
    deflation = ShiftedDeflation(problem, power=1, shift=1)

    dc = DeflatedContinuation(problem=problem, deflation=deflation, teamsize=1, verbose=True, clear_output=True, profile=True)
    dc.run(values={"f": 0})

    if backend.comm_world.rank == 0:
        print()
        params = (0,)
        branches = dc.thread.io.known_branches(params)
        for branch in branches:
            functionals = dc.thread.io.fetch_functionals([params], branch)[0]
            print("Solution: %s" % functionals)
