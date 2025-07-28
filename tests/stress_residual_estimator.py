from firedrake import *
import numpy
import firedrake.functionspaceimpl as impl
from ufl.core.ufl_type import ufl_type


nx = 30
mesh = UnitSquareMesh(nx, nx)
degree = 1

# Define function spaces
S = VectorFunctionSpace(mesh, "BDM", degree)
V = VectorFunctionSpace(mesh, "DG", degree-1)
Q = FunctionSpace(mesh, "CG", degree)

# Mixed Function Space
T = S * V * Q

print(S.value_shape, S.value_size)
print(V.value_shape, V.value_size)
print(Q.value_shape, Q.value_size)
print(T.value_shape, T.value_size)

# Mixed test function
test = TestFunction(T)
t = Function(T)

# symbolic split
sigma, u, gamma = split(t)
tau, v, eta = split(test)

dim = mesh.geometric_dimension()

mu = Constant(1)
lam = Constant(100)
I = Identity(dim)

def A(sigma):
    return (1/(2*mu)) * (sigma - (lam/(2*mu+dim*lam)) * tr(sigma)*I)
def Ainv(sigma):
    return (2*mu*sigma + lam * tr(sigma)*I)
def skw(sigma):
    return sigma[0, 1] - sigma[1, 0]

x, y = SpatialCoordinate(mesh)

uexact = as_vector([x*y*sin(pi*y), 0])
sigma_exact = Ainv(sym(grad(uexact)))
g = div(sigma_exact)

u0 = uexact
n = FacetNormal(mesh)

# F =〈Aσ, τ 〉 + 〈div σ, v〉 + 〈u, div τ 〉 + 〈σ, η〉 + 〈γ, τ〉 -〈g, v〉 - 〈u0, τ · n〉∂Ω
F = (inner(A(sigma), tau)*dx
     + inner(u, div(tau))*dx
     + inner(gamma, skw(tau))*dx
     + inner(div(sigma), v)*dx
     + inner(skw(sigma), eta)*dx
     - inner(g, v)*dx
     - inner(u0, dot(tau, n))*ds
     )

#bcs = [DirichletBC(T.sub(0), 0, (3,4))]
bcs = []
solve(F == 0, t, bcs=bcs)

# Goal Functional
psi = y * (y-1)
M = inner(dot(sigma, n), as_vector([psi, 0]))*ds(2)
M_calculated = assemble(M)
M_exact = assemble(replace(M, {sigma: sigma_exact}))

print("Calculated functional, M = ", M_calculated)
print("Exact functional, M = ", M_exact)

error_actual = abs(M_calculated - M_exact)
print("Goal error = ", error_actual)
# numerical split
sigma, u, gamma = t.subfunctions
sigma.rename("sigma")
u.rename("u")
gamma.rename("gamma")
print("error u", errornorm(uexact, u))
print("error sigma", errornorm(sigma_exact, sigma))
print("error skw(sigma)", norm(skw(sigma)))
file = VTKFile("output/stress.pvd")
file.write(sigma, u, gamma)

# Solve the dual problem (MANUALLY)
element = T.ufl_element()
degree = element.degree()
dual_degree = degree + 1
dual_element = PMGPC.reconstruct_degree(element, dual_degree)
T_dual = FunctionSpace(mesh, dual_element)

z = Function(T_dual)
z_trial = TrialFunction(T_dual)
z_test = TestFunction(T_dual)

G = action(adjoint(derivative(F, t, z_trial)), z) - derivative(M, t, z_test)
bcs_dual  = [bc.reconstruct(V=T_dual, indices=bc._indices, g=0) for bc in bcs]
solve(G == 0, z, bcs_dual) # Obtain z

def residual(F, test):
    v = F.arguments()[0]
    return replace(F, {v: test})

etah = assemble(residual(F, z))
print("Global error estimator: ", etah)


z_lo = Function(T, name="LowOrderDualSolution")
z_lo.interpolate(z)
z_err = z - z_lo

# Automatic computation
cell = mesh.ufl_cell()  #Returns the cell from the mesh
dim = mesh.topological_dimension() # Dimension of the mesh
variant = "integral" # Finite element type
residual_degree = degree
residual_sp = {"snes_type": "ksponly",
               "ksp_type": "preonly",
               "pc_type": "hypre",
               "pc_hypre_type": "pilut"}

# ---------------- Equation 4.6 to find cell residual Rcell -------------------------
B = FunctionSpace(mesh, "B", dim+1, variant=variant) # Bubble function space
bubbles = Function(B).assign(1) # Bubbles

# Discontinuous function space of Rcell polynomials
DG = TensorFunctionSpace(mesh, "DG", residual_degree, variant=variant, shape=T.value_shape)
uc = TrialFunction(DG)
vc = TestFunction(DG)
ac = inner(uc, bubbles*vc)*dx
Lc = residual(F, bubbles*vc)

Rcell = Function(DG, name="Rcell") # Rcell polynomial
print("Computing Rcells ...")
solve(ac == Lc, Rcell, solver_parameters=residual_sp) # solve for Rcell polynonmial

def both(u):
    return u("+") + u("-")

# ---------------- Equation 4.8 to find facet residual Rfacet -------------------------
FB = FunctionSpace(mesh, "FB", dim, variant=variant) # Cone function space
cones = Function(FB).assign(1) # Cones

el = BrokenElement(FiniteElement("FB", cell=cell, degree=residual_degree+dim, variant=variant))
Q = TensorFunctionSpace(mesh, el, shape=T.value_shape)
q = TestFunction(Q)
p = TrialFunction(Q)
Lf = residual(F, q) - inner(Rcell, q)*dx
af = both(inner(p/cones, q))*dS + inner(p/cones, q)*ds

Rhat = Function(Q)
print("Computing Rhats ...")
solve(af == Lf, Rhat, solver_parameters=residual_sp)
Rfacet = Rhat/cones

# 8. Compute error indicators eta_T
DG0 = FunctionSpace(mesh, "DG", degree=0)
test = TestFunction(DG0)
vol = CellVolume(mesh)
etaT = Function(DG0)

#eta_T = assemble((inner(test*Rcell, z_err)*dx +  avg(inner(test*Rfacet,z_err))*dS + inner(test*Rfacet,z_err)*ds))

G = (
    inner(etaT / vol, test)*dx
    - inner(inner(Rcell, z_err), test)*dx +
    - inner(avg(inner(Rfacet,z_err)), both(test))*dS +
    - inner(inner(Rfacet,z_err), test)*ds
    )

print("Computing eta_T indicators ...")
sp = {"mat_type": "matfree", "ksp_type": "richardson", "pc_type": "jacobi"}
solve(G == 0, etaT, solver_parameters=sp)

with etaT.dat.vec as evec:
        evec.abs()
        etaT_array = evec.getArray()

etaT_total = np.sum(etaT_array)
print(f"sum_T(eta_T): {etaT_total}")

# Compute efficiency indices
eff1 = etah/error_actual
eff2 = etaT_total/error_actual
print(f"Efficiency index 1 = {eff1}")
print(f"Efficiency index 2 = {eff2}")


exit()

dim = 2

variant = "integral"
B = FunctionSpace(mesh, "B", dim+1, variant=variant)

R = VectorFunctionSpace(mesh, "DG", degree, dim=T.value_size)

element = BrokenElement(FiniteElement("FB", degree+dim))
