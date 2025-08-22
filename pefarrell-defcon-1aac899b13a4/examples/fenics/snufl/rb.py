# -*- coding: utf-8 -*-
import sys
from   math import floor

from defcon import *
from dolfin import *

params = {
    "snes_max_it": 100,
    "snes_atol": 1.0e-9,
    "snes_rtol": 0.0,
    "snes_monitor": None,
    "snes_converged_reason": None,
    "ksp_type": "fgmres",
    "ksp_converged_reason": None,
    "ksp_monitor": None,
    "ksp_gmres_restart": 200,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "multiplicative",
    "pc_fieldsplit_0_fields": "0,1",
    "pc_fieldsplit_1_fields": "2",
    "pc_fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "lu",
    "fieldsplit_0_pc_factor_mat_solver_package": "mumps",
    "fieldsplit_0_pc_factor_mat_solver_type": "mumps",
    "fieldsplit_1_ksp_type": "preonly",
    "fieldsplit_1_pc_type": "lu"
}


mesh = RectangleMesh(Point(0, 0), Point(5, 1), 100, 100)

Ve = VectorElement("CG", triangle, 2)
Qe = FiniteElement("CG", triangle, 1)
Te = FiniteElement("CG", triangle, 1)
Ze = MixedElement([Ve, Qe, Te])
Z  = FunctionSpace(mesh, Ze)

Ra = Constant(1705)
Pr = Constant(6.8)

z = Function(Z)
w = TestFunction(Z)

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

bcs = [
    DirichletBC(Z.sub(0), (0, 0), "on_boundary"),
    DirichletBC(Z.sub(2), 1, "near(x[1], 0.0)"),
    DirichletBC(Z.sub(2), 0, "near(x[1], 1.0)"),
    DirichletBC(Z.sub(1), 0, "x[0] == 0.0 && x[1] == 0.0", "pointwise")
]

prob = nonlinearproblem.GeneralProblem(F, z, bcs)
solver = nonlinearsolver.SNUFLSolver(prob, solver_parameters=params)

solver.solve()
