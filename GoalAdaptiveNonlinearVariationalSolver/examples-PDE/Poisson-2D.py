from firedrake import *
from netgen.occ import *
import sys
from goal_adaptivity import GoalAdaptiveNonlinearVariationalSolver
from goal_adaptivity import getlabels

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
        print("named: ", f.center.x)
        f.name = 'goal_face'
        print(f.name)

geo = OCCGeometry(shape, dim = 2)
ngmesh = geo.GenerateMesh(maxh=initial_mesh_size)
mesh = Mesh(ngmesh)

mesh = Mesh(unit_square.GenerateMesh(maxh=initial_mesh_size))

# Define solver parameters ---------------------
solver_parameters = {
    "max_iterations": 20,
    "output_dir": "output/conv-diff-new",
    #"uniform_refinement": True
    #"use_adjoint_residual": True
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

sp_dual = {"snes_type": "ksponly",
            "ksp_type": "cg",
            "ksp_rtol": 1.0e-3,
            "ksp_max_it": 5,
            "ksp_convergence_test": "skip",
            "ksp_monitor": None,
            "pc_type": "python",
            "pc_python_type": "firedrake.ASMStarPC",
            "pc_star_mat_ordering_type": "metisnd",
            "pc_star_sub_sub_pc_type": "cholesky"
            }

sp_primal = {"pc_type": "cholesky",
            "pc_factor_mat_solver_type": "mumps"}

problem = NonlinearVariationalProblem(F, u, bcs)

GoalAdaptiveNonlinearVariationalSolver(problem, J, tolerance, solver_parameters, primal_solver_parameters=sp_primal,
                                       dual_solver_parameters=sp_dual, 
                                       exact_solution=u_exact).solve()
