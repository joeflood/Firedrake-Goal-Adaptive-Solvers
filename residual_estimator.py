from firedrake import *


nx = 5  #Mesh size
degree = 1 # Solution polynomial degree
variant = "integral" # Finite element type 

mesh = UnitSquareMesh(nx, nx) #Create unit square mesh

cell = mesh.ufl_cell()  #Returns the cell from the mesh
dim = mesh.topological_dimension() # Dimension of the mesh 

V = FunctionSpace(mesh, "CG", degree, variant=variant)
# PDE residual
uh = Function(V, name="solution")
v = TestFunction(V)

# MMS Method of Manufactured Solution
x, y = SpatialCoordinate(mesh)
uexact = sin(pi*x) * cos(pi*y)
f = - div(grad(uexact))
#f = 1

F = inner(grad(uh), grad(v))*dx - inner(f, v)*dx

bcs = DirichletBC(V, uexact, "on_boundary")

solve(F == 0, uh, bcs=bcs)


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

dual_space = FunctionSpace(mesh, "Lagrange", degree) #Dual function space
u_dual = TrialFunction(dual_space) # Symbolic trial function to differentiate F
v_dual = TestFunction(dual_space) # Dual test function
z = Function(dual_space) # Dual soluton

F_dual = inner(grad(z), grad(v_dual))*dx - inner(f, v_dual)*dx
bilinear_form = derivative(F_dual, z)
bilinear_form_adj = adjoint(bilinear_form)
bcs_dual  = DirichletBC(dual_space, 0.0, "on_boundary")

# Define goal functional
ds = Measure("ds", domain=mesh)  # Boundary measure
n = FacetNormal(mesh)
# -------- Goal functional options: --------
# J = inner(grad(v_dual), n) * ds # Boundary flux
J = v_dual * dx
solve(bilinear_form_adj == J, z, bcs_dual) 

dual_space_high = FunctionSpace(mesh, "Lagrange", degree + 1)
u_dual_high = TrialFunction(dual_space_high) # Symbolic trial function to differentiate F
v_dual_high = TestFunction(dual_space_high) # Dual test function
z_high = Function(dual_space_high) # Dual soluton
J = v_dual_high * dx

F_dual_high = inner(grad(z_high), grad(v_dual_high))*dx - inner(f, v_dual_high)*dx
bilinear_form_high = derivative(F_dual_high, z_high)
bilinear_form_adj_high = adjoint(bilinear_form_high)
bcs_dual_high  = DirichletBC(dual_space_high, 0.0, "on_boundary")
solve(bilinear_form_adj_high == J, z_high, bcs_dual_high)


#z_high = Function(dual_space_high).interpolate(z)
#zh = Function(dual_space).interpolate(z)
zerr = z

print("z      :", z.dat.data)        # writable view
print("z_high :", z_high.dat.data)


eta = assemble(inner(test*Rcell, zerr)*dx +  avg(inner(test*Rfacet,zerr))*dS + inner(test*Rfacet,zerr)*ds)
with eta.dat.vec as evec:
    evec.abs()

print(eta.dat.data)

total_eta = np.sum(eta.dat.data)
print("Total error estimator:", total_eta)

eta_z = assemble(inner(test*Rcell, z_high)*dx +  avg(inner(test*Rfacet,z_high))*dS + inner(test*Rfacet,z_high)*ds)
with eta_z.dat.vec as evec_z:
    evec_z.abs()

print(eta_z.dat.data)

total_eta_z = np.sum(eta_z.dat.data)
print("Total error estimator:", total_eta_z)


# Plot
DG = FunctionSpace(mesh, "DG", degree=degree)
Rfacet_plot = Function(DG, name="Rfacet").interpolate(Rhat/cones)

uerr = Function(DG, name="error").interpolate(uh - uexact)
file = VTKFile("output/solution.pvd")
file.write(uh, uerr, Rcell, Rfacet_plot)
