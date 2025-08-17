from firedrake import *
from netgen.occ import *
import sys
from goal_adaptivity import GoalAdaptiveNonlinearVariationalSolver

nx = 10
mesh = Mesh(unit_square.GenerateMesh(maxh=1/nx))
degree = 1

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
    "output_dir": "output/nonlinear",
    "write_at_iteration": True
}

# Define actual problem -----------------------
n = FacetNormal(mesh)
V = FunctionSpace(mesh, "CG", degree, variant="integral") # Template function space used to define the PDE
u = Function(V, name="Solution")
v = TestFunction(V)
(x, y) = SpatialCoordinate(u.function_space().mesh()) # MMS Method of Manufactured Solution
u_exact = sin(pi*x)*sin(pi*y)
f = -div(grad(u_exact)) + u_exact**3

F = inner(grad(u), grad(v))*dx + u**3*v*dx- inner(f, v)*dx
bcs = [DirichletBC(V, u_exact, "on_boundary")]

J = dot(grad(u), n)*ds(2)
tolerance = 0.000001

problem = NonlinearVariationalProblem(F, u, bcs)
GoalAdaptiveNonlinearVariationalSolver(problem, J, tolerance, solver_parameters, u_exact).solve()


