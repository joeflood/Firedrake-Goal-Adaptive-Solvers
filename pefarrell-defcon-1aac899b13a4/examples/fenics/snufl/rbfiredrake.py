# -*- coding: utf-8 -*-
import sys
from   math import floor

#from defcon import *
from firedrake import *
import numpy

params = {"snes_max_it": 100,
          "snes_atol": 1.0e-9,
          "snes_rtol": 0.0,
          "snes_monitor": None,
          "snes_converged_reason": None,
          "ksp_type": "fgmres",
          "ksp_converged_reason": None,
          "ksp_monitor": None,
          "ksp_gmres_restart": 200,
          "mat_type": "aij",
          "pc_type": "fieldsplit",
          "pc_fieldsplit_type": "multiplicative",
          "pc_fieldsplit_0_fields": "0,1",
          "pc_fieldsplit_1_fields": "2",
          "pc_fieldsplit_0_ksp_type": "preonly",
          "fieldsplit_0_pc_type": "lu",
          "fieldsplit_0_pc_factor_mat_solver_package": "mumps",
          "fieldsplit_0_pc_factor_mat_solver_type": "mumps",
          "fieldsplit_1_ksp_type": "preonly",
          "fieldsplit_1_pc_type": "lu",
          "ksp_view": None,
}

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


mesh = RectangleMesh(50, 50, 5, 1)

V = VectorFunctionSpace(mesh, "CG", 2)
W = FunctionSpace(mesh, "CG", 1)
Z  = V*W*W

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
    DirichletBC(Z.sub(0), Constant((0, 0)), (1, 2, 3, 4)),
    DirichletBC(Z.sub(2), Constant(1.0), (3,)),
    DirichletBC(Z.sub(2), Constant(0.0), (4,)),
    PointwiseBC(Z.sub(1), Constant(0.0), "near(x[0], 0.0) && near(x[1], 0.0)")
]


prob = NonlinearVariationalProblem(F, z, bcs)
solver = NonlinearVariationalSolver(prob, solver_parameters=params)

solver.solve()
