from firedrake import *
from netgen.occ import *
import sys
from algorithm import *
from ufl.algorithms.analysis import extract_constants

initial_mesh_size = 0.2

box1 = WorkPlane().MoveTo(0, 0).Rectangle(4, 1).Face()
box2 = WorkPlane().MoveTo(1.4, 0).Rectangle(0.2, 0.5).Face()

# Now they are geometric shapes you can combine
shape = box1 - box2

tol = 0.001
for f in shape.edges: # Assign face labels
    print(f.center.x)
    if abs(f.center.x - 4) < tol:
        print("Outflow named.")
        f.name = "outflow"
    elif abs(f.center.x) < tol:
        print("Inflow named.")
        f.name = "inflow"
    else:
        print("Dirichlet named")
        f.name = "dirichlet"

geo = OCCGeometry(shape, dim = 2)
ngmesh = geo.GenerateMesh(maxh=initial_mesh_size)
mesh = Mesh(ngmesh)
meshctx = MeshCtx(mesh)

# Define solver parameters ---------------------
solver_parameters = {
    "degree": 1,
    "dual_solve_method": "high_order",
    "dual_solve_degree": "degree + 1",
    "residual_solve_method": "automatic",
    "residual_degree": "degree",
    "dorfler_alpha": 0.5,
    "goal_tolerance": 0.00001,
    "max_iterations": 10,
    "output_dir": "output/navierstokes",
    "parameter_init": 20,
    "parameter_final": 0.02,
    "parameter_iterations": 10,
    "write_at_iteration": True
}

solverctx = SolverCtx(solver_parameters)

# Define actual problem -----------------------
def define_problem(meshctx: MeshCtx, solverctx: SolverCtx):
    mesh = meshctx.mesh

    # Define function spaces
    V = VectorFunctionSpace(mesh, "CG", degree=solverctx.degree+1, dim=2)
    P = FunctionSpace(mesh, "CG", degree=solverctx.degree)
    T = V * P
    test = TestFunction(T)
    t = Function(T)

    # symbolic split
    u, p = split(t)
    v, q = split(test)

    x, y = SpatialCoordinate(mesh)

    nu = Constant(2, name="nu") # Viscosity
    p_inflow = 1
    p_outflow = 0
    n = meshctx.n

    labels = meshctx.labels
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

    # Goal Functional
    M = dot(u,n) * ds_outflow
    M_exact = 0.40863917 # Exact solution given in Rognes & Logg ex.3
    
    return ProblemCtx(space=T, trial=t, test=test, residual=F, bcs=bcs, goal=M, goal_exact=M_exact, parameter=nu)

adaptive_problem = GoalAdaptionStabilized(meshctx, define_problem, solverctx)

adaptive_problem.solve()
