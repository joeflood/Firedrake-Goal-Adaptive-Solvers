from firedrake import *
from netgen.occ import *
import sys
from algorithm import *

# Define initial mesh ---------------------
initial_mesh_size = 0.2

box1 = WorkPlane().MoveTo(-1, 0).Rectangle(1, 1).Face()
box2 = WorkPlane().MoveTo(0, 0).Rectangle(1, 1).Face()
box3 = WorkPlane().MoveTo(0, -1).Rectangle(1, 1).Face()

# Now they are geometric shapes you can combine
shape = box1 + box2 + box3

tol = 0.00001
for f in shape.edges: # Assign face labels
    if abs(f.center.x + 1) < tol:
        f.name = "goal_face"

geo = OCCGeometry(shape, dim = 2)
ngmesh = geo.GenerateMesh(maxh=initial_mesh_size)
mesh = Mesh(ngmesh)

mesh = Mesh(unit_square.GenerateMesh(maxh=initial_mesh_size))

meshctx = MeshCtx(mesh)

# Define solver parameters ---------------------
solver_parameters = {
    "degree": 1,
    "dual_solve_method": "high_order",
    "dual_solve_degree": "degree + 1",
    "residual_solve_method": "automatic",
    "residual_degree": "degree",
    "dorfler_alpha": 0.5,
    "goal_tolerance": 0.000001,
    "max_iterations": 30,
    "output_dir": "output/poisson2d",
    "write_at_iteration": True
}

solverctx = SolverCtx(solver_parameters)

# Define actual problem -----------------------
def define_problem(meshctx: MeshCtx, solverctx: SolverCtx):
    mesh = meshctx.mesh
    V = FunctionSpace(mesh, "CG", solverctx.degree, variant="integral") # Template function space used to define the PDE
    u = Function(V, name="Solution")
    v = TestFunction(V)
    (x, y) = SpatialCoordinate(u.function_space().mesh()) # MMS Method of Manufactured Solution
    u_exact = (x-1)*(y-1)**2
    f = -div(grad(u_exact))
    g = dot(grad(u_exact), meshctx.n) 

    labels = meshctx.labels
    #ds_goal = Measure("ds", domain=mesh, subdomain_id=labels['goal_face'])
    dxm     = Measure("dx", domain=mesh)

    F = inner(grad(u), grad(v))*dxm - inner(f, v)*dxm
    bcs = [DirichletBC(V, u_exact, "on_boundary")]

    J = dot(grad(u), meshctx.n) * ds

    return ProblemCtx(space=V, trial=u, test=v, exact=u_exact, residual=F, bcs=bcs, goal=J, f=f, g=g)

adaptive_problem = GoalAdaption(meshctx, define_problem, solverctx)

adaptive_problem.solve()
