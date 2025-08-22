# -*- coding: utf-8 -*-
import sys
from   math import floor

from petsc4py import PETSc
from firedrake import *
from defcon import *

import numpy

params = {
    "snes_max_it": 100,
    "snes_atol": 1.0e-9,
    "snes_rtol": 0.0,
    "snes_monitor": None,
    "snes_linesearch_type": "basic",
    "mat_type": "aij",
    "ksp_type": "preonly",
    "ksp_converged_reason": None,
    "ksp_monitor": None,
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps"
}

options = PETSc.Options()
for k in params:
    options[k] = params[k]
parameters["matnest"] = False
#parameters["default_matrix_type"] = "aij"

# This hack enforces the boundary condition at (0, 0)
class PointwiseBC(DirichletBC):
    @utils.cached_property
    def nodes(self):
        x = self.function_space().mesh().coordinates.dat.data_ro
        zero = numpy.array([0, 0])
        dists = [numpy.linalg.norm(pt - zero) for pt in x]
        minpt = numpy.argmin(dists)
        if dists[minpt] < 1.0e-10:
            out = numpy.array([minpt], dtype=numpy.int32)
        else:
            out = numpy.array([], dtype=numpy.int32)
        return out

class RayleighBenardProblem(BifurcationProblem):
    def mesh(self, comm):
        return RectangleMesh(50, 50, 5.0, 1.0, comm=comm)

    def function_space(self, mesh):
        V = VectorFunctionSpace(mesh, "CG", 2)
        W = FunctionSpace(mesh, "CG", 1)
        T = FunctionSpace(mesh, "CG", 1)
        return MixedFunctionSpace([V, W, T])

    def parameters(self):
        Ra = Constant(0)
        Pr = Constant(0)
        return [
                (Ra, "Ra", r"$\mathrm{Ra}$"),
                (Pr, "Pr", r"$\mathrm{Pr}$")
               ]

    def residual(self, z, params, w):
        (Ra, Pr)  = params
        (u, p, T) = split(z)
        (v, q, S) = split(w)

        g = as_vector([0, 1])

        F = (
            inner(grad(u), grad(v))*dx
            + inner(dot(grad(u), u), v)*dx
            - inner(p, div(v))*dx
            - Ra*Pr*inner(T*g, v)*dx
            + inner(div(u), q)*dx
            + inner(dot(grad(T), u), S)*dx
            + 1/Pr * inner(grad(T), grad(S))*dx
        )

        return F

    def boundary_conditions(self, Z, params):
        bcs = [
                DirichletBC(Z.sub(0), Constant((0, 0)), (1, 2, 3, 4)),
                DirichletBC(Z.sub(2), Constant(1.0), (3,)),
                DirichletBC(Z.sub(2), Constant(0.0), (4,)),
                PointwiseBC(Z.sub(1), Constant(0.0), "near(x[0], 0.0) && near(x[1], 0.0)")
              ]
        return bcs

    def functionals(self):
        def sqL2(z, params):
            (u, p, T) = split(z)
            j = assemble(inner(u, u)*dx)
            return j

        return [(sqL2, "sqL2", r"$\|u\|^2$")]

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, Z, params, n):
        x = SpatialCoordinate(Z.mesh())
        z = Function(Z)
        (u, p, T) = z.split()
        u.interpolate(as_vector([0.09*+sin(4*pi*x[0])*sin(3*pi*x[1]), 0.17*-sin(5.5*pi*x[0])*sin(2*pi*x[1])]))
        p.interpolate(5800*x[1])
        T.interpolate(1 - x[1])

        return z

    def number_solutions(self, params):
        (Ra, Pr) = params
        if Ra < 1700:
            return 1
        if Ra < 1720:
            return 3
        return float("inf")

    def squared_norm(self, z, w, params):
        (zu, zp, zT) = split(z)
        (wu, wp, wT) = split(w)
        diffu = zu - wu
        diffp = zp - wp
        diffT = zT - wT
        return inner(diffu, diffu)*dx + inner(grad(diffu), grad(diffu))*dx + inner(diffp, diffp)*dx + inner(diffT, diffT)*dx

    def save_pvd(self, z, pvd, params):
        (u, p, T) = z.split()
        u.rename("Velocity", "Velocity")
        p.rename("Pressure", "Pressure")
        T.rename("Temperature", "Temperature")
        pvd.write(u, p, T)

if __name__ == "__main__":
    dc = DeflatedContinuation(problem=RayleighBenardProblem(), teamsize=1, verbose=True)
    dc.run(values={"Ra": [1701], "Pr": [6.8]}, freeparam="Ra")
