from firedrake import *
from netgen.occ import *
import sys
from goal_adaptivity import GoalAdaptiveNonlinearVariationalSolver
from goal_adaptivity import getlabels

# Define initial mesh ---------------------
initial_mesh_size = 0.2

# Initial mesh
box1 = Box(Pnt(-1,0,-1), Pnt(0,1,0))
box2 = Box(Pnt(0,0,-1), Pnt(1,1,0))
box3 = Box(Pnt(0,-1,-1), Pnt(1,0,0))
shape = box1 + box2 + box3

tol = 0.00000001
for f in shape.faces: # Assign face labels
    if abs(f.center.x + 1) < tol:
        f.name = "goal_face"
    elif abs(f.center.x - 1) < tol or abs(f.center.y - 1) < tol:
        f.name = "dirichletbcs"
    else: 
        f.name = "neumannbcs"  

geo = OCCGeometry(shape)
ngmesh = geo.GenerateMesh(maxh=initial_mesh_size)
mesh = Mesh(ngmesh)


# Define solver parameters ---------------------
solver_parameters = {
    "degree": 1,
    "dual_solve_method": "star",
    "dual_solve_degree": "degree + 1",
    "residual_solve_method": "automatic",
    "residual_degree": "degree",
    "dorfler_alpha": 0.5,
    "max_iterations": 30,
    "output_dir": "output/poisson3d",
    "write_at_iteration": True
}

degree = 1
n = FacetNormal(mesh)
V = FunctionSpace(mesh, "CG", degree, variant="integral") # Template function space used to define the PDE
u = Function(V, name="Solution")
v = TestFunction(V)
(x, y, z) = SpatialCoordinate(u.function_space().mesh()) # MMS Method of Manufactured Solution
u_exact = (x-1)*(y-1)**2
G = as_vector(((y-1)**2, 2*(x-1)*(y-1), 0.0))
g = dot(G,n)
f = -div(grad(u_exact))

labels = getlabels(mesh)
ds_goal = Measure("ds", domain=mesh, subdomain_id=labels['goal_face'])
dxm     = Measure("dx", domain=mesh)
ds_neumann     = Measure("ds", domain=mesh, subdomain_id=labels['neumannbcs']+labels['goal_face'])
ds_dirichlet = Measure("ds", domain=mesh, subdomain_id=labels['dirichletbcs'])

F = inner(grad(u), grad(v))*dxm - inner(f, v)*dxm - g*v*ds_neumann
bcs = [DirichletBC(V, u_exact, labels['dirichletbcs'])]

J = u*ds_goal
tolerance = 0.00001

sp_dual = {"snes_type": "ksponly",
            "ksp_type": "cg",
            "ksp_rtol": 1.0e-10,
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
                                       dual_solver_parameters=sp_dual, exact_solution=u_exact).solve()
