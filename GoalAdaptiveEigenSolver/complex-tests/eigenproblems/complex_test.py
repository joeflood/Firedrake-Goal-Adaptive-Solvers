import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

from firedrake import *
from firedrake.eigensolver import LinearEigenproblem, LinearEigensolver
from netgen.occ import *
import numpy as np
from firedrake.__future__ import interpolate as interp  # symbolic interpolate
import sys

NEV_SOLVE = 15
degree = 1
nx = 20
# Mesh and spaces
mesh = Mesh(unit_square.GenerateMesh(maxh=1/nx))

V  = FunctionSpace(mesh, "CG", degree)
u  = TrialFunction(V); v = TestFunction(V)
A  = inner(grad((u)), grad(v)) * dx
M  = inner((u), v)* dx
bcs = [DirichletBC(V, 0.0, "on_boundary")]



prob = LinearEigenproblem(A, M, bcs=bcs, restrict=True)
es = LinearEigensolver(prob, n_evals=NEV_SOLVE)
nconv = es.solve()
lam, vecs = [], []
for i in range(min(nconv, NEV_SOLVE)):
    lam.append(es.eigenvalue(i))
    vr, vi = es.eigenfunction(i)
    vecs.append(vr)