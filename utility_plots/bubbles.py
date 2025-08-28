from firedrake import *

degree = 5
mesh = UnitTriangleMesh()

nx = 3
mesh = UnitSquareMesh(nx, nx)

x = SpatialCoordinate(mesh)
a = cos(pi/3)
b = sin(pi/3)
F = as_matrix([[1, a], [0, b]])
#mesh.coordinates.interpolate(F*x)


B = FunctionSpace(mesh, "Bubble",  degree+1, variant="integral")
bubble = Function(B, name="bubbles")

f = VTKFile("output/bubble.pvd")

bubble.assign(1)
f.write(bubble, time=0)

dim = B.dim()
for k in range(dim):
    bubble.assign(0)
    bubble.dat.data[k] = 1
    f.write(bubble, time=k+1)



element = FiniteElement("FB", degree=degree, variant="integral")
element = BrokenElement(element)
FB = FunctionSpace(mesh, element)
cone = Function(FB, name="cones")

f = VTKFile("output/facetbubble.pvd")

cone.assign(1)
f.write(cone, time=0)

dim = FB.dim()
for k in range(dim):
    cone.assign(0)
    cone.dat.data[k] = 1
    f.write(cone, time=k+1)
