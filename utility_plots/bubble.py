from firedrake import *

mesh = UnitSquareMesh(3, 3)

V = FunctionSpace(mesh, "Bubble", mesh.topological_dimension() + 1)
b = Function(V)
b.dat.data[:] = 1
VTKFile("output/bubble.pvd").write(b)