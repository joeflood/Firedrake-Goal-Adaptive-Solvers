# -*- coding: utf-8 -*-
import sys

# This test a simple field-split preconditioner in dolfin
# using a made-up system of two coupled diffusion problems.

from dolfin import *
from defcon import nonlinearsolver, nonlinearproblem

solver_params = {
    "snes_type": "ksponly",
    "ksp_type": "gmres",
    "ksp_monitor": None,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "additive",
    "fieldsplit_0_fields": "0",
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "lu",
    "fieldsplit_1_fields": "1",
    "fieldsplit_1_ksp_type": "preonly",
    "fieldsplit_1_pc_type": "lu",
}

mesh = UnitSquareMesh(100, 100)

E1 = FiniteElement("CG", triangle, 1)
E2 = FiniteElement("CG", triangle, 2)

Z  = FunctionSpace(mesh, E1*E2)

z = Function(Z)
w = TestFunction(Z)

u1, u2 = split(z)
v1, v2 = split(w)

f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)

F = (
    inner(grad(u1), grad(v1))*dx + u1*v1*dx + u1*v2*dx
    + inner(grad(u2), grad(v2))*dx + u2*v2*dx - u2*v1*dx
    -f*v1*dx
    )

bcs = None


prob = nonlinearproblem.GeneralProblem(F, z, bcs)
solver = nonlinearsolver.SNUFLSolver(prob, solver_parameters=solver_params)

solver.solve()
