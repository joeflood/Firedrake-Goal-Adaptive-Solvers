# -*- coding: utf-8 -*-
import sys
from   math import floor

from firedrake import *

# solves a simple system of two coupled diffusion problems in firedrake
# with a simple fieldsplit preconditioner.

params = {"snes_type": "ksponly",
          "ksp_type": "gmres",
          "ksp_monitor": None,
          "ksp_view": None,
          "pc_type": "fieldsplit",
          "pc_fieldsplit_type": "additive",
          "fieldsplit_0_fields": 0,
          "fieldsplit_1_fields": 1,
          "fieldsplit_0_ksp_type": "preonly",
          "fieldsplit_0_pc_type": "lu",
          "fieldsplit_1_ksp_type": "preonly",
          "fieldsplit_1_pc_type": "lu"}

mesh = UnitSquareMesh(2, 2)

E1 = FunctionSpace(mesh, "CG", 1)
E2 = FunctionSpace(mesh, "CG", 2)

Z  = MixedFunctionSpace([E1, E2])

z = Function(Z)
w = TestFunction(Z)

u1, u2 = split(z)
v1, v2 = split(w)

f = interpolate(Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2), E1)

F = (
    inner(grad(u1), grad(v1))*dx + u1*v1*dx
    + inner(grad(u2), grad(v2))*dx + u2*v2*dx
    -f*v1*dx - f*v2*dx
    )

bcs = None


prob = NonlinearVariationalProblem(F, z, bcs)
solver = NonlinearVariationalSolver(prob, solver_parameters=params)

solver.solve()
