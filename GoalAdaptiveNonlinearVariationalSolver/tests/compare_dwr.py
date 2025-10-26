from firedrake import *


mesh = UnitSquareMesh(10, 10, quadrilateral=False)
degree = 1
V = FunctionSpace(mesh, "CG", degree)
print("DOF = ", V.dim())

# PDE residual
u = Function(V, name="Solution")
v = TestFunction(V)

# MMS Method of Manufactured Solution
(x, y) = SpatialCoordinate(u.function_space().mesh())
u_exact = (x**2*y + 3*x*y**2)
f = -div(grad(u_exact)) + u_exact**3

F = inner(grad(u), grad(v))*dx + u**3*v*dx - inner(f, v)*dx
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

sp_chol = {"pc_type": "cholesky",
           "pc_factor_mat_solver_type": "mumps"}
sp_pmg = {"snes_type": "ksponly",
          "ksp_type": "cg",
          "ksp_rtol": 1.0e-10,
          "ksp_max_it": 1,
          "ksp_convergence_test": "skip",
          "ksp_monitor": None,
          "pc_type": "python",
          "pc_python_type": "firedrake.P1PC",
          "pmg_mg_coarse": {
              "pc_type": "python",
              "pc_python_type": "firedrake.AssembledPC",
              "assembled_pc_type": "cholesky",
          },
          "pmg_mg_levels": {
              "ksp_max_it": 1,
              "ksp_type": "chebyshev",
              "pc_type": "python",
              "pc_python_type": "firedrake.ASMStarPC",
              "pc_star_mat_ordering_type": "metisnd",
              "pc_star_sub_sub_pc_type": "cholesky",
          }
      }
sp_star = {"snes_type": "ksponly",
          "ksp_type": "cg",
          "ksp_rtol": 1.0e-10,
          "ksp_max_it": 10,
          "ksp_convergence_test": "skip",
          "ksp_monitor": None,
          "pc_type": "python",
          "pc_python_type": "firedrake.ASMStarPC",
          "pc_star_mat_ordering_type": "metisnd",
          "pc_star_sub_sub_pc_type": "cholesky",
          }

solve(G == 0, z, bcs_dual, solver_parameters=sp_chol) # Obtain z
z_lo = Function(V).interpolate(z)
z_err = z - z_lo


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

#print("Local error estimators: ")
#print(eta.dat.data)

print(f"Sum of local error estimators (Rannacher): {sum(eta.dat.data)}")

# Insert automatic computation:

cell = mesh.ufl_cell()  #Returns the cell from the mesh
dim = mesh.topological_dimension() # Dimension of the mesh 
variant = "integral" # Finite element type 

# ---------------- Equation 4.6 to find cell residual Rcell -------------------------
residual_degree = degree + 1

B = FunctionSpace(mesh, "B", dim+1, variant=variant) # Bubble function space
bubbles = Function(B).assign(1) # Bubbles

# Discontinuous function space of Rcell polynomials
DG = FunctionSpace(mesh, "DG", residual_degree, variant=variant)
uc = TrialFunction(DG)
vc = TestFunction(DG)
ac = inner(uc, bubbles*vc)*dx
Lc = residual(F, bubbles*vc)

Rcell = Function(DG, name="Rcell") # Rcell polynomial
cell_sp     = {"mat_type": "matfree",
               "snes_type": "ksponly",
               "ksp_type": "preonly",
               "pc_type": "python",
               "pc_python_type": "firedrake.PatchPC",
               "patch_pc_patch_save_operators": True,
               "patch_pc_patch_construct_type": "vanka",
               "patch_pc_patch_construct_codim": 0,
               "patch_pc_patch_sub_mat_type": "seqdense",
               "patch_sub_ksp_type": "preonly",
               "patch_sub_pc_type": "lu",
              }

solve(ac == Lc, Rcell, solver_parameters=cell_sp) # solve for Rcell polynonmial

def both(u):
    return u("+") + u("-")

# ---------------- Equation 4.8 to find facet residual Rfacet -------------------------
FB = FunctionSpace(mesh, "FB", dim, variant=variant) # Cone function space
cones = Function(FB).assign(1) # Cones

el = BrokenElement(FiniteElement("FB", cell=cell, degree=residual_degree+dim, variant=variant))
Q = FunctionSpace(mesh, el)
q = TestFunction(Q)
p = TrialFunction(Q)
Lf = residual(F, q) - inner(Rcell, q)*dx
af = both(inner(p/cones, q))*dS + inner(p/cones, q)*ds

Rhat = Function(Q)
facet_sp    = {"mat_type": "matfree",
               "snes_type": "ksponly",
               "ksp_type": "cg",
               "ksp_monitor_true_residual": None,
               "pc_type": "jacobi",
               "pc_hypre_type": "pilut"}
solve(af == Lf, Rhat, solver_parameters=facet_sp)
Rfacet = Rhat/cones

# Extra code - another way of accomplishing the same outcome?
#el = BrokenElement(FiniteElement("DGT", cell=cell, degree=residual_degree, variant=variant))
#DGT = FunctionSpace(mesh, el)
#Rfacet = Function(DGT).interpolate(Rhat/cones)


L2_Rcell = sqrt(assemble(inner(Rcell, Rcell) * dx))
print("‖Rcell‖_L2(Ω) =", L2_Rcell)

# Rfacet L2 norms:
# boundary L2
L2_boundary = sqrt(assemble(inner(Rfacet, Rfacet)*ds))
print("‖Rfacet‖_L2(boundary) =", L2_boundary)

# interior facet L2 (average)
L2_interior = sqrt(assemble(inner(avg(Rfacet), avg(Rfacet))*dS))
print("‖Rfacet‖_L2(interior) =", L2_interior)
DG0 = FunctionSpace(mesh, "DG", degree=0)
test = TestFunction(DG0)

eta_T = assemble(
    inner(inner(Rcell, z_err), test)*dx + 
    + inner(avg(inner(Rfacet, z_err)), both(test))*dS + 
    + inner(inner(Rfacet, z_err), test)*ds
)

#eta_T = assemble(inner(test*Rcell, z_err)*dx +  avg(inner(test*Rfacet,z_err))*dS + inner(test*Rfacet,z_err)*ds)

# eta_T = Function(DG0)
# G = (
#      inner(eta_T / vol, test)*dx
#      - inner(inner(Rcell, z_err), test)*dx + 
#      - inner(avg(inner(Rfacet,z_err)), both(test))*dS + 
#      - inner(inner(Rfacet,z_err), test)*ds
#     )

# sp = {"mat_type": "matfree", "ksp_type": "richardson", "pc_type": "jacobi"}
# solve(G == 0, eta_T, solver_parameters=sp)


with eta_T.dat.vec as evec:
    evec.abs()

total_eta = np.sum(eta_T.dat.data)
print("Automatic total error estimator:", total_eta)

#factor = eta_T.dat.data / eta.dat.data
#print("Factor relative to local method:")
#print(factor)

# Exact facet residuals
eta_manual = Function(DG0)
n = FacetNormal(mesh)
H = (
    inner(eta_manual / vol, test)*dx
     - inner(f + div(grad(u)), z_err * test) * dx
     - 0.5 * inner(jump(-grad(u), n), z_err * test('+')) * dS
     - 0.5 * inner(jump(-grad(u), n), z_err * test('-')) * dS
     - inner(dot(-grad(u), n), z_err * test) * ds
)

# Each cell is an independent 1x1 solve, so Jacobi is exact
sp = {"mat_type": "matfree", "ksp_type": "richardson", "pc_type": "jacobi"}
solve(H == 0, eta_manual, solver_parameters=sp)

with eta_manual.dat.vec as evec:
    evec.abs()

#print(eta_manual.dat.data)
manual_total = np.sum(eta_manual.dat.data)
print("Manual total error estimator: ", manual_total)

difference = (eta_manual.dat.data - eta_T.dat.data)

total_difference = np.sum(difference)
print("Total difference:", total_difference)
