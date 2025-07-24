from firedrake import *
from netgen.occ import *
import sys
print("Interpreter:", sys.executable)
from algorithm import *

# Define initial mesh ---------------------
initial_mesh_size = 0.2

# Initial mesh
box1 = Box(Pnt(-1,0,-1), Pnt(0,1,0))
box2 = Box(Pnt(0,0,-1), Pnt(1,1,0))
box3 = Box(Pnt(0,-1,-1), Pnt(1,0,0))
shape = box1 + box2 + box3

for f in shape.faces: # Assign face labels
    if f.center.x == -1:
        f.name = "goal_face"
    if f.center.x == 1 or f.center.y == 1:
        f.name = "dirichletbcs"

geo = OCCGeometry(shape)
ngmesh = geo.GenerateMesh(maxh=initial_mesh_size)
mesh = Mesh(ngmesh)
VTKFile("adaptivemesh_unrefined.pvd").write(mesh)

meshctx = MeshCtx(mesh)

# Define solver parameters ---------------------
solver_parameters = {
    "degree": 1,
    "dual_solve_method": "high_order",
    "dual_solve_degree": "degree + 1",
    "residual_solve_method": "automatic",
    "residual_degree": "degree",
    "dorfler_alpha": 0.5,
    "goal_tolerance": 0.001,
    "max_iterations": 30,
    "output_dir": "../output/poisson3d"
}

solverctx = SolverCtx(solver_parameters)

# Define actual problem -----------------------
def define_problem(meshctx: MeshCtx, solverctx: SolverCtx):
    mesh = meshctx.mesh
    V = FunctionSpace(mesh, "CG", solverctx.degree, variant="integral") # Template function space used to define the PDE
    u = Function(V, name="Solution")
    v = TestFunction(V)
    (x, y, z) = SpatialCoordinate(u.function_space().mesh()) # MMS Method of Manufactured Solution
    u_exact = (x-1)*(y-1)**2
    G = as_vector(((y-1)**2, 2*(x-1)*(y-1), 0.0))
    g = dot(G,meshctx.n)
    f = -div(grad(u_exact))

    labels = meshctx.labels
    ds_goal = Measure("ds", domain=mesh, subdomain_id=labels['goal_face'])
    dxm     = Measure("dx", domain=mesh)
    dsm     = Measure("ds", domain=mesh)

    F = inner(grad(u), grad(v))*dxm - inner(f, v)*dxm - g*v*dsm
    bcs = [DirichletBC(V, u_exact, labels['dirichletbcs'])]

    J = dot(grad(u), meshctx.n)*ds_goal

    return ProblemCtx(V, u, v, u_exact, F, bcs, J)

adaptive_problem = GoalAdaption(meshctx, define_problem, solverctx)

adaptive_problem.solve()
