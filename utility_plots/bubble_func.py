from firedrake import *

mesh = UnitSquareMesh(8, 8, quadrilateral=False)

#   λ_i
W1     = VectorFunctionSpace(mesh, "Lagrange", 1)
λ      = TestFunction(W1)

#   bubble  b_T
Wb     = FunctionSpace(mesh, "Bubble", 1)
b_T    = TestFunction(Wb)

#   cone β_{T}^{S}  (facet opposite vertex 0)
β      = λ[1]*λ[2]

# verify that b_T vanishes on boundary:
assert assemble(b_T*ds) < 1e-14

# verify that β is nonzero only on that facet:
print("∫_boundary β ds =", assemble(β*ds))
