from firedrake import *
from netgen.occ import *
import sys
from goal_adaptivity import GoalAdaptiveNonlinearVariationalSolver
from goal_adaptivity import getlabels

# Define initial mesh ---------------------
initial_mesh_size = 0.1
mesh = Mesh(unit_square.GenerateMesh(maxh=initial_mesh_size))

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

degree = 1
# Define actual problem -----------------------
n = FacetNormal(mesh)
V = FunctionSpace(mesh, "CG", degree, variant="integral") # Template function space used to define the PDE
u = Function(V, name="Solution")
v = TestFunction(V)
(x, y) = SpatialCoordinate(mesh) # MMS Method of Manufactured Solution
u_exact = (x**2*y + 3*x*y**2)
f = -div(grad(u_exact))

F = inner(grad(u), grad(v))*dx - inner(f, v)*dx
bcs = [DirichletBC(V, u_exact, "on_boundary")]

J = dot(grad(u), n) * ds
tolerance = 0.0001

problem = NonlinearVariationalProblem(F, u, bcs)

GoalAdaptiveNonlinearVariationalSolver(problem, J, tolerance, solver_parameters, u_exact).solve()
