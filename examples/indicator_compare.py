from firedrake import *
from netgen.occ import *
import sys
from algorithm import *

nx = 10
mesh = Mesh(unit_square.GenerateMesh(maxh=1/nx))
degree = 1

meshctx = MeshCtx(mesh)

# Define solver parameters ---------------------
solver_parameters = {
    "degree": 1,
    "dual_solve_method": "high_order",
    "dual_solve_degree": "degree + 1",
    "residual_solve_method": "automatic",
    "residual_degree": "degree",
    "dorfler_alpha": 0.5,
    "goal_tolerance": 0.0001,
    "max_iterations": 12,
    "output_dir": "output/nonlinear"
}

solverctx = SolverCtx(solver_parameters)

# Define actual problem -----------------------
def define_problem(meshctx: MeshCtx, solverctx: SolverCtx):
    mesh = meshctx.mesh
    V = FunctionSpace(mesh, "CG", solverctx.degree, variant="integral") # Template function space used to define the PDE
    u = Function(V, name="Solution")
    v = TestFunction(V)

    
    (x, y) = SpatialCoordinate(u.function_space().mesh())
    u_exact = (x**2*y + 3*x*y**2)
    f = -div(grad(u_exact))

    F = inner(grad(u), grad(v))*dx - inner(f, v)*dx
    bcs = [DirichletBC(V, u_exact, "on_boundary")]
    
    J = dot(grad(u), meshctx.n)*ds
    #J = dot(grad(u), meshctx.n)*ds(2)
    g = Constant(0)
    ds_neumann = ds()

    return ProblemCtx(V, u, v, u_exact, F, bcs, J, g, f, ds_neumann)


adaptive_problem = GoalAdaption(meshctx, define_problem, solverctx)

adaptive_problem.solve()



