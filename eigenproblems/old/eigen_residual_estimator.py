from firedrake import *


nx = 10  #Mesh size
degree = 1 # Solution polynomial degree
variant = "integral" # Finite element type 

mesh = UnitSquareMesh(nx, nx) #Create unit square mesh

cell = mesh.ufl_cell()  #Returns the cell from the mesh
dim = mesh.topological_dimension() # Dimension of the mesh 

V = FunctionSpace(mesh, "CG", degree, variant=variant)
# PDE residual
uh = Function(V, name="solution")
du = TrialFunction(V)
v = TestFunction(V)

# MMS Method of Manufactured Solution
x, y = SpatialCoordinate(mesh)
uexact = 256*(1-x)*x*(1-y)*y*exp(-((x-0.5)**2+(y-0.5)**2)/10)
f = - div(grad(uexact))

F = inner(grad(uh), grad(v))*dx - inner(f, v)*dx

bcs = [DirichletBC(V, uexact, "on_boundary")]

solve(F == 0, uh, bcs=bcs)

n = FacetNormal(mesh)
J = dot(grad(uh), n)*ds

def residual(F, test):
    v = F.arguments()[0]
    return replace(F, {v: test})

# ---------------- Equation 4.6 to find cell residual Rcell -------------------------
B = FunctionSpace(mesh, "B", dim+1, variant=variant)
bubbles = Function(B).assign(1)

DG = FunctionSpace(mesh, "DG", degree, variant=variant)
uc = TrialFunction(DG)
vc = TestFunction(DG)
ac = inner(uc, bubbles*vc)*dx
Lc = residual(F, bubbles*vc)

Rcell = Function(DG, name="Rcell")
solve(ac == Lc, Rcell)

def both(u):
    return u("+") + u("-")

# ---------------- Equation 4.8 to find facet residual Rfacet -------------------------
FB = FunctionSpace(mesh, "FB", dim, variant=variant)
cones = Function(FB).assign(1)

el = BrokenElement(FiniteElement("FB", cell=cell, degree=degree+dim, variant=variant))
Q = FunctionSpace(mesh, el)
q = TestFunction(Q)
p = TrialFunction(Q)
Lf = residual(F, q) - inner(Rcell, q)*dx
af = both(inner(p/cones, q))*dS + inner(p/cones, q)*ds

Rhat = Function(Q)
solve(af == Lf, Rhat)


el = BrokenElement(FiniteElement("DGT", cell=cell, degree=degree, variant=variant))
DGT = FunctionSpace(mesh, el)
#Rfacet = Function(DGT).interpolate(Rhat/cones)
Rfacet = Rhat/cones



DG0 = FunctionSpace(mesh, "DG", degree=0)
test = TestFunction(DG0)

# ========================== solve the dual problem ==========================

# Solve dual in degree + 1
dual_space = FunctionSpace(mesh, "Lagrange", degree + 1) #Dual function space
u_dual = TrialFunction(dual_space) # Symbolic trial function to differentiate F
v_dual = TestFunction(dual_space) # Dual test function
z = Function(dual_space) # Dual soluton

G = action(adjoint(derivative(F, uh, du)), z) - derivative(J, uh, v_dual)
G = replace(G, {v: v_dual})
bcs_dual  = [bc.reconstruct(V=dual_space, g=0) for bc in bcs]

# Define goal functional
ds = Measure("ds", domain=mesh)  # Boundary measure
n = FacetNormal(mesh)
# Goal functional options:
# J = inner(grad(v_dual), n) * ds # Boundary flux

solve(G == 0, z, bcs_dual) # Obtain z

Juh = assemble(J)
Ju = assemble(replace(J, {uh: uexact}))
print(f"Global error estimator: {assemble(residual(F, z))}")
print(f"J(u): {Ju} J(uh) = {Juh} J(uh) - J(u) = {Juh - Ju}")

dual_space_low = FunctionSpace(mesh, "Lagrange", degree) #Dual function space
z_h = Function(dual_space_low).interpolate(z)
zerr = z #- z_h

eta = assemble(inner(test*Rcell, zerr)*dx +  avg(inner(test*Rfacet,zerr))*dS + inner(test*Rfacet,zerr)*ds)
with eta.dat.vec as evec:
    evec.abs()

print(eta.dat.data)

total_eta = np.sum(eta.dat.data)
print("Total error estimator:", total_eta)

# Plot
DG = FunctionSpace(mesh, "DG", degree=degree)
Rfacet_plot = Function(DG, name="Rfacet").interpolate(Rhat/cones)

uerr = Function(DG, name="error").interpolate(uh - uexact)
file = VTKFile("output/solution.pvd")
file.write(uh, uerr, Rcell, Rfacet_plot)
