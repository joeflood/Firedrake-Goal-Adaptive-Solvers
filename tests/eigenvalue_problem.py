from firedrake import *
from netgen.occ import *

nx = 10
mesh = Mesh(unit_square.GenerateMesh(maxh=1/nx))
degree = 1

# Define actual problem -----------------------
V = FunctionSpace(mesh, "CG", degree) # Template function space used to define the PDE
u = Function(V, name="Solution")
v = TestFunction(V)
coords = SpatialCoordinate(u.function_space().mesh()) 
x, y = coords[0], coords[1]

# MMS Method of Manufactured Solution
u_exact = sin(pi*x)*sin(pi*y)
lambda_exact = 2*pi**2

A = inner(grad(u), grad(v)) * dx
M = inner(u,v) * dx

bcs = [DirichletBC(V, u_exact, "on_boundary")]

eigenproblem = LinearEigenproblem(A, M, bcs)

