from firedrake import *
from netgen.occ import *

nx = 10  #Mesh size
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

F = inner(grad(uh), grad(v))*dx - inner(f, v)*dx

bcs = DirichletBC(V, uexact, "on_boundary")

tol = 0.1

# From here we repeat
# 2. Compute the finite element solution of the primal problem on the current mesh.
solve(F == 0, uh, bcs=bcs)

def residual(F, test):
    v = F.arguments()[0]
    return replace(F, {v: test})

# 3. & 4. Solve the dual problem
dual_space = FunctionSpace(mesh, "Lagrange", degree + 1) #Dual function space W_h (+1)
u_dual = TrialFunction(dual_space) # Symbolic trial function to differentiate F
v_dual = TestFunction(dual_space) # Dual test function
z = Function(dual_space) # Dual soluton

F_dual = inner(grad(z), grad(v_dual))*dx - inner(f, v_dual)*dx
bilinear_form = derivative(F_dual, z)
bilinear_form_adj = adjoint(bilinear_form)
bcs_dual  = DirichletBC(dual_space, 0.0, "on_boundary") # Use homogenize instead

# Define goal functional
ds = Measure("ds", domain=mesh)  # Boundary measure
n = FacetNormal(mesh)
# Goal functional options:
# J = inner(grad(v_dual), n) * ds # Boundary flux
J = v_dual * dx

solve(bilinear_form_adj == J, z, bcs_dual) # Obtain z

dual_space_low = FunctionSpace(mesh, "Lagrange", degree) #Dual function space W_h
z_h = Function(dual_space_low).interpolate(z)
zerr = z - z_h

# 5. & 6. Compute eta_h to determine whether to continue.
r_form = residual(F, z)      # == replace(F, {v : z})
eta_h     = abs(assemble(r_form))   # scalar: r(z)
print(eta_h)

# 7. Compute cell and facet residuals R_T, R_\partialT
# Equation 4.6 to find cell residual Rcell
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

# Equation 4.8 to find facet residual Rfacet
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

# 8. Compute error indicators eta_T 
DG0 = FunctionSpace(mesh, "DG", degree=0)
test = TestFunction(DG0)
#eta = assemble(inner(test*Rcell, zerr)*dx +  avg(inner(test*Rfacet,zerr))*dS + inner(test*Rfacet,zerr)*ds)

eta = Function(DG0)
G = - inner(eta, test) + inner(test*Rcell, zerr)*dx +  avg(inner(test*Rfacet,zerr))*dS + inner(test*Rfacet,zerr)*ds
sp = {"mat_type": "matfree", "ksp_type": "richardson", "pc_type": "jacobi"}
solve(G == 0, eta, solver_parameters=sp)

with eta.dat.vec as evec:
    evec.abs()
print(eta.dat.data)
total_eta = np.sum(eta.dat.data)
print("Total error estimator:", total_eta)

# 9. Mark cells for refinement 



# 10. Refine cells marked for refinement

# Repeat



# Plot
