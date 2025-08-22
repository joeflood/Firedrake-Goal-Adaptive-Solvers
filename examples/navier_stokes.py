from firedrake import *
from netgen.occ import *
import sys
from goal_adaptivity import GoalAdaptiveNonlinearVariationalSolver
from ufl.algorithms.analysis import extract_constants
from goal_adaptivity import getlabels

def l2_norm(f):
    return assemble(inner(f, f)*dx)**0.5

initial_mesh_size = 0.01

box1 = WorkPlane().MoveTo(0, 0).Rectangle(4, 1).Face()
box2 = WorkPlane().MoveTo(1.4, 0).Rectangle(0.2, 0.5).Face()

# Now they are geometric shapes you can combine
shape = box1 - box2

tol = 0.01
for f in shape.edges: # Assign face labels
    if abs(f.center.x - 4) < tol:
        f.name = "outflow"
    elif abs(f.center.x) < tol:
        f.name = "inflow"
    else:
        f.name = "dirichlet"

geo = OCCGeometry(shape, dim = 2)
ngmesh = geo.GenerateMesh(maxh=initial_mesh_size)
mesh = Mesh(ngmesh)

# Define solver parameters ---------------------
solver_parameters = {
    "max_iterations": 20,
    "output_dir": "output/conv-diff-new",
    #"uniform_refinement": True
    #"use_adjoint_residual": True
}
# Define actual problem -----------------------
degree = 1
# Define function spaces
V = VectorFunctionSpace(mesh, "CG", degree=degree+1, dim=2)
P = FunctionSpace(mesh, "CG", degree=degree)
T = V * P
test = TestFunction(T)
t = Function(T)

# symbolic split
u, p = split(t)
v, q = split(test)

x, y = SpatialCoordinate(mesh)

nu = Constant(0.02, name="nu") # Viscosity
p_inflow = 1
p_outflow = 0
n = FacetNormal(mesh)

labels = getlabels(mesh,1)
ds_outflow = Measure("ds", domain=mesh, subdomain_id=labels['outflow'])
ds_inflow = Measure("ds", domain=mesh, subdomain_id=labels['inflow'])

F = (
    nu * inner(grad(u), grad(v)) * dx +
    inner(dot(u, grad(u)),v) * dx -
    inner(p, div(v)) * dx +
    inner(div(u), q) * dx +
    inner(p_outflow * n, v) * ds_outflow +
    inner(p_inflow * n, v) * ds_inflow
)

bcs = [DirichletBC(T.sub(0), 0, sub_domain=labels['dirichlet'])]

sp_primal = {"snes_monitor": None,
             "snes_linesearch_monitor": None,
             "snes_linesearch_type": "l2"}

problem = NonlinearVariationalProblem(F, t, bcs)
nls = NonlinearVariationalSolver(problem, solver_parameters=sp_primal)

# Goal Functional
M = dot(u,n) * ds_outflow
M_exact = 0.40863917 # Exact solution given in Rognes & Logg ex.3
tolerance = 0.000001

# Parameter continuation loop for initial guess
visc_schedule = np.logspace(np.log10(0.05), np.log10(0.02), num=20)

for i, nu_val in enumerate(visc_schedule):
    nu.assign(float(nu_val))        # update viscosity Constant in-place
    if i == 0:
        t.assign(0.0)               # start from zero on the first step
    print(f"[continuation] {i+1}/{len(visc_schedule)} | nu = {float(nu_val):.6g}")
    nls.solve()                     # uses previous 't' as the initial guess for the next step
    u_norm = l2_norm(u)
    p_norm = l2_norm(p)
    print("l2 norm of u:", u_norm)
    print("l2 norm of p:", p_norm)


adaptive_problem = GoalAdaptiveNonlinearVariationalSolver(problem,  M, tolerance, solver_parameters, exact_goal=M_exact, primal_solver_parameters=sp_primal)
adaptive_problem.solve()
