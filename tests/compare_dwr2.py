from firedrake import *
from netgen.occ import *

initial_mesh_size = 0.5
# Initial mesh
box1 = WorkPlane().MoveTo(-1, 0).Rectangle(1, 1).Face()
box2 = WorkPlane().MoveTo(0, 0).Rectangle(1, 1).Face()
box3 = WorkPlane().MoveTo(0, -1).Rectangle(1, 1).Face()

# Now they are geometric shapes you can combine
shape = box1 + box2 + box3

for f in shape.edges: # Assign face labels
    if f.center.x == -1:
        f.name = "goal_face"
    elif f.center.x == 1 or f.center.y == 1:
        f.name = "dirichletbcs"
    else:
        f.name = "neumannbcs"

geo = OCCGeometry(shape, dim = 2)
ngmesh = geo.GenerateMesh(maxh=initial_mesh_size)
mesh = Mesh(ngmesh)
VTKFile("adaptivemesh_unrefined.pvd").write(mesh)

# Solver parameters
degree = 1
max_iterations = 30
dual_solve_method = "high_order" # Options: high_order, star
#residual_solve_method = "automatic" # Options: manual, automatic
residual_degree = degree + 15 # Degree of residuals 

# Output vectors
N_vec = []
eta_vec = []
etah_vec = []
etaTsum_vec = []
eff1_vec = []
eff2_vec = []

def boundary_labels(mesh):
    ngmesh = mesh.netgen_mesh
    names = ngmesh.GetRegionNames(codim=1)
    names_to_labels = {}
    for l in names:
        names_to_labels[l] = tuple(i+1 for i, name in enumerate(names) if name == l)
    return names_to_labels

def residual(F, test): # Residual helper function
    v = F.arguments()[0]
    return replace(F, {v: test})

def build_primal_problem(mesh): # Define PDE problem & Goal Functional
    V = FunctionSpace(mesh, "CG", degree, variant="integral") # Template function space used to define the PDE
    u = Function(V, name="Solution")
    v = TestFunction(V)
    coords = SpatialCoordinate(u.function_space().mesh()) # MMS Method of Manufactured Solution
    x, y = coords[0], coords[1]
    u_exact = (x-1)**5*(y-1)**6*sin(3*pi*x*y)**3
    G = as_vector(((y-1)**2, 2*(x-1)*(y-1)))
    n = FacetNormal(mesh)
    g = dot(G,n)
    f = -div(grad(u_exact))

    labels = boundary_labels(mesh)
    ds_goal = Measure("ds", domain=mesh, subdomain_id=labels['goal_face'])

    F = inner(grad(u), grad(v))*dx - inner(f, v)*dx
    bcs = [DirichletBC(V, u_exact, "on_boundary")]
    
    J = dot(grad(u), n)*ds_goal
    data = dict(V=V, u=u, v=v, u_exact=u_exact, f=f, n=n, J_form=J, bcs=bcs, F=F, g=g)
    return data


def residual(F, test):
    v = F.arguments()[0]
    return replace(F, {v: test})

prob = build_primal_problem(mesh)
V      = prob["V"]
u      = prob["u"]
f      = prob["f"]
g      = prob["g"]
v      = prob["v"]
u_exact= prob["u_exact"]
F      = prob["F"]
bcs    = prob["bcs"]
J = prob["J_form"]
n = FacetNormal(mesh)
labels = boundary_labels(mesh)
ds_neumann = Measure("ds", domain=mesh, subdomain_id=labels['goal_face'] + labels['neumannbcs'])
ds_dirichlet = Measure("ds", domain=mesh, subdomain_id=labels['dirichletbcs'])

ndofs = V.dim()
print("N:" , ndofs)

# Solve dual in degree + 1
Vf = FunctionSpace(mesh, "Lagrange", degree + 1) #Dual function space
vz = TestFunction(Vf) # Dual test function
z = Function(Vf) # Dual soluton

ndofs = Vf.dim()
print("Ndual:" , ndofs)

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
B = FunctionSpace(mesh, "B", dim + 1, variant=variant) # Bubble function space
bubbles = Function(B).assign(1) # Bubbles

# Discontinuous function space of Rcell polynomials
DG = FunctionSpace(mesh, "DG", degree=residual_degree, variant=variant)
uc = TrialFunction(DG)
vc = TestFunction(DG)
ac = inner(uc, bubbles*vc)*dx
Lc = residual(F, bubbles*vc)

Rcell = Function(DG, name="Rcell") # Rcell polynomial
solve(ac == Lc, Rcell) # solve for Rcell polynonmial

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
solve(af == Lf, Rhat)
# Rfacet = Rhat/cones

# Extra code - another way of accomplishing the same outcome?
el = BrokenElement(FiniteElement("DGT", cell=cell, degree=residual_degree, variant=variant))
DGT = FunctionSpace(mesh, el)
Rfacet = Function(DGT).interpolate(Rhat/cones)


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

#eta_T = assemble(inner(test*Rcell, z_err)*dx +  avg(inner(test*Rfacet,z_err))*dS + inner(test*Rfacet,z_err)*ds)

eta_T = Function(DG0)
G = (
     inner(eta_T / vol, test)*dx
     - inner(inner(Rcell, z_err), test)*dx + 
     - inner(avg(inner(Rfacet,z_err)), both(test))*dS + 
     - inner(inner(Rfacet,z_err), test)*ds
 )

sp = {"mat_type": "matfree", "ksp_type": "richardson", "pc_type": "jacobi"}
solve(G == 0, eta_T, solver_parameters=sp)

with eta_T.dat.vec as evec:
    evec.abs()

#print("Automatically computed local error estimates:")
#print(eta_T.dat.data)

total_eta = np.sum(eta_T.dat.data)
print("Automatic total error estimator:", total_eta)

factor = eta_T.dat.data / eta.dat.data
#print("Factor relative to local method:")
#print(factor)


# Exact facet residuals
eta_manual = Function(DG0)

n = FacetNormal(mesh)

H = (
    inner(eta_manual / vol, test)*dx
     - inner(f + div(grad(u)), z_err * test) * dx
     + 0.5 * inner(jump(grad(u), n), z_err * test('+')) * dS
     + 0.5 * inner(jump(grad(u), n), z_err * test('-')) * dS
     + inner(dot(grad(u), n) + g, z_err * test) * ds_neumann
     + inner(dot(grad(u), n), z_err * test) * ds_dirichlet
)

# Each cell is an independent 1x1 solve, so Jacobi is exact
sp = {"mat_type": "matfree", "ksp_type": "richardson", "pc_type": "jacobi"}
solve(H == 0, eta_manual, solver_parameters=sp)

with eta_manual.dat.vec as evec:
    evec.abs()

#print("Manually computed local error estimates:")
#print(eta_manual.dat.data)

manual_total = np.sum(eta_manual.dat.data)
print("Manual total error estimator: ", manual_total)

difference = (eta_manual.dat.data - eta_T.dat.data)
#print("Difference:")
#print(difference)

total_difference = np.sum(difference)
print("Total difference:", total_difference)
