from firedrake import *


mesh = UnitSquareMesh(8, 8, quadrilateral=True)
degree = 1
V = FunctionSpace(mesh, "CG", degree, variant="integral")

# PDE residual
u = Function(V, name="Solution")
v = TestFunction(V)

# MMS Method of Manufactured Solution
(x, y) = SpatialCoordinate(u.function_space().mesh())
u_exact = 256*(1-x)*x*(1-y)*y*exp(-((x-0.5)**2+(y-0.5)**2)/10)
f = -div(grad(u_exact))

F = inner(grad(u), grad(v))*dx - inner(f, v)*dx
bcs = [DirichletBC(V, u_exact, "on_boundary")]
solve(F == 0, u, bcs=bcs)

n = FacetNormal(mesh)
J = dot(grad(u), n)*ds
#J = inner(u, u)*dx

def residual(F, test):
    v = F.arguments()[0]
    return replace(F, {v: test})

# ========================== solve the dual problem ==========================

# Solve dual in degree + 1
Vf = FunctionSpace(mesh, "Lagrange", degree + 1) #Dual function space
vz = TestFunction(Vf) # Dual test function
z = Function(Vf) # Dual soluton

G = action(adjoint(derivative(F, u, TrialFunction(Vf))), z) - derivative(J, u, vz)
G = replace(G, {v: vz})
bcs_dual  = [bc.reconstruct(V=Vf, g=0) for bc in bcs]

solve(G == 0, z, bcs_dual) # Obtain z

Juh = assemble(J)
Ju = assemble(replace(J, {u: u_exact}))
print(f"Global error estimator: {assemble(residual(F, z))}")
print(f"J(u): {Ju} J(uh) = {Juh} J(uh) - J(u) = {Juh - Ju}")
