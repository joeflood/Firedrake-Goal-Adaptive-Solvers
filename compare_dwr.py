from firedrake import *


mesh = UnitSquareMesh(5, 5, quadrilateral=False)
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

# Now localise by hand. (3.18) in Becker & Rannacher
W = FunctionSpace(mesh, "DG", 0)
w = TestFunction(W)
rho = Function(W)
omega = Function(W)
h = CellDiameter(mesh)
vol = CellVolume(mesh)

R = f + div(grad(u))
r = 0.5 * jump(grad(u), n)
both = lambda u: u("+") + u("-")
Rho = (
        inner(rho / vol, w)*dx
      - inner(sqrt(inner(R, R)), w)*dx
      - (both(h)**(-0.5) * inner(sqrt(inner(r, r)), both(w)))*dS
      )
dgsp = {"mat_type": "matfree", "ksp_type": "richardson", "pc_type": "jacobi"}
solve(Rho == 0, rho, solver_parameters=dgsp)

z_lo = Function(V, name="LowOrderDualSolution")
z_lo.interpolate(z)
z_err = z - z_lo
Omega = (
          inner(omega / vol, w)*dx
        - inner(sqrt(inner(z_err, z_err)), w)*dx
        - (both(h)**(+0.5) * inner(sqrt(inner(z_err, z_err)), both(w)))*dS
        )
solve(Omega == 0, omega, solver_parameters=dgsp)

eta = Function(W, name="LocalErrorEstimate")
eta.interpolate(rho*omega)

print("Local error estimators: ")
print(eta.dat.data)

print(f"Sum of local error estimators: {sum(eta.dat.data)}")
