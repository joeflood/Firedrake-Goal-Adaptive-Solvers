from firedrake import *
from netgen.occ import *
import sys
from goal_adaptivity import GoalAdaptiveNonlinearVariationalSolver
from ufl.algorithms.analysis import extract_constants
from goal_adaptivity import getlabels

def l2_norm(f):
    return assemble(inner(f, f)*dx)**0.5

initial_mesh_size = 0.05

box1 = WorkPlane().MoveTo(0, 0).Rectangle(2.2, 0.41).Face()
circle = WorkPlane().MoveTo(0.2, 0.2).Circle(0.05).Face()

# Now they are geometric shapes you can combine
shape = box1 - circle

tol = 0.01
for f in shape.edges: # Assign face labels
    if abs(f.center.x - 2.2) < tol:
        f.name = "outflow"
    elif abs(f.center.x) < tol:
        f.name = "inflow"
    elif abs(f.center.x - 0.2) < tol:
        f.name = "cylinder"
    else:
        f.name = "noslip"

geo = OCCGeometry(shape, dim = 2)
ngmesh = geo.GenerateMesh(maxh=initial_mesh_size)
mesh = Mesh(ngmesh)

for f in shape.edges: # Assign face labels
    print(f.name)

# Define solver parameters ---------------------
solver_parameters = {
    "max_iterations": 20,
    "run_name": "cylinder_lift",
    "output_dir": "output",
    #"uniform_refinement": True
    #"use_adjoint_residual": True
}
# Define actual problem -----------------------
degree = 2
# Define function spaces
V = VectorFunctionSpace(mesh, "CG", degree=degree, dim=2)
P = FunctionSpace(mesh, "CG", degree=degree-1)
T = V * P
test = TestFunction(T)
t = Function(T)

# symbolic split
u, p = split(t)
v, q = split(test)

x, y = SpatialCoordinate(mesh)

nu = Constant(0.001, name="nu") # Viscosity
H = Constant(0.41)
uin = as_vector([0.3 * (4.0 * y * (H - y) / H**2), 0.0])
labels = getlabels(mesh,1)
ds_in   = ds(labels["inflow"])
ds_out  = ds(labels["outflow"])
ds_wall = ds(labels["noslip"])
ds_cyl  = ds(labels["cylinder"])
n = FacetNormal(mesh)

# Traditional form
F = (
    nu*inner(grad(u), grad(v))*dx                      # ν ∫ ∇u:∇v
  + inner(dot(u, nabla_grad(u)), v)*dx                # ∫ (u·∇)u · v
  - p*div(v)*dx
  + q*div(u)*dx
)

bcs = [
    DirichletBC(T.sub(0), uin,       labels["inflow"]),   # inflow velocity
    DirichletBC(T.sub(0), Constant((0, 0)),
                [labels["noslip"], labels["cylinder"]]),  # walls + cylinder
    # No Dirichlet on outflow (natural traction-free)
]

# --- pressure gauge: mean-zero pressure nullspace ---
nullspace = MixedVectorSpaceBasis(T, [T.sub(0), VectorSpaceBasis(constant=True)])

# Goal: Lift around cylinder
I = Identity(2)
traction_on_body = (nu*grad(u) - p*I)*n 
e2 = as_vector([0.0, 1.0])
M = 500 *dot(traction_on_body, e2)*ds_cyl

M_exact = 0.010618948146 # Exact solution given in Rognes & Logg ex.3
tolerance = 0.00000001

sp_primal = {"snes_monitor": None,
             "snes_linesearch_monitor": None,
             "snes_linesearch_type": "l2"}

problem = NonlinearVariationalProblem(F, t, bcs)
nls = NonlinearVariationalSolver(problem, solver_parameters=sp_primal)

# Parameter continuation loop for initial guess
visc_schedule = np.logspace(np.log10(0.05), np.log10(0.001), num=5)

pvd = VTKFile("output/navier_stokes.pvd")
t.subfunctions[0].rename("Velocity")
t.subfunctions[1].rename("Pressure")
for i, nu_val in enumerate(visc_schedule):
    nu.assign(float(nu_val))        # update viscosity Constant in-place
    if i == 0:
        t.assign(0.0)               # start from zero on the first step
    print(f"[continuation] {i+1}/{len(visc_schedule)} | nu = {float(nu_val):.6g} dim = {t.function_space().dim()}")
    nls.solve()                     # uses previous 't' as the initial guess for the next step
    pvd.write(t.subfunctions[0], t.subfunctions[1], time=float(nu_val))
    u_norm = l2_norm(u)
    p_norm = l2_norm(p)
    print("l2 norm of u:", u_norm)
    print("l2 norm of p:", p_norm)
    e1 = as_vector([1.0, 0.0])
    Umean_meas = assemble(dot(u, e1) * ds_in) / H
    print("  Umean_meas ~", float(Umean_meas))
    Juh = assemble(500*dot(dot(nu*grad(u) - p*I, n), e2) * ds_cyl)
    print("J(u_h) = ", Juh)

print(nu)
problem = NonlinearVariationalProblem(F, t, bcs)
adaptive_problem = GoalAdaptiveNonlinearVariationalSolver(problem,  M, tolerance, solver_parameters, primal_solver_parameters=sp_primal, nullspace=nullspace)
adaptive_problem.solve()
