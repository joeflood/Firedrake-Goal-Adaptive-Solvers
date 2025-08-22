from firedrake import *
from netgen.occ import *
import sys
from goal_adaptivity import GoalAdaptiveNonlinearVariationalSolver

def boundary_labels(mesh):
    ngmesh = mesh.netgen_mesh
    names = ngmesh.GetRegionNames(codim=1)
    names_to_labels = {}
    for l in names:
        names_to_labels[l] = tuple(i+1 for i, name in enumerate(names) if name == l)
    return names_to_labels

# Define initial mesh ---------------------
initial_mesh_size = 1

outer = WorkPlane().MoveTo(0, 0).Rectangle(5, 5).Face()
roi   = WorkPlane().MoveTo(3, 3).Rectangle(1, 1).Face()
roi.name = "roi"
ring = outer - roi
ring.name = "bulk"
shape = Glue([ring, roi])

tol = 0.00001
for f in shape.edges: # Assign face labels
    if abs(f.center.x) < tol:
        f.name = 'dirichletbc'
    else:
        f.name = 'neumannbc'

geo = OCCGeometry(shape, dim=2)
ngmesh = geo.GenerateMesh(maxh=initial_mesh_size)
mesh = Mesh(ngmesh)

labels = boundary_labels(mesh)
ds_neumann = Measure("ds", domain=mesh, subdomain_id=labels['neumannbc'])
ds_dirichlet = Measure("ds", domain=mesh, subdomain_id=labels['dirichletbc'])
names = mesh.netgen_mesh.GetRegionNames(codim=0)
name_to_id = {name: i+1 for i, name in enumerate(names)}
ROI = name_to_id["roi"]
dx_goal = Measure("dx", domain=mesh, subdomain_id=ROI)

lam = Constant(3.0)   # will homotopy 0 → target (e.g. 10)
def reaction(w): return lam*exp(w)

degree = 1
V = FunctionSpace(mesh, "CG", degree) # Template function space used to define the PDE
u = Function(V, name="Solution")
v = TestFunction(V)
(x, y) = SpatialCoordinate(u.function_space().mesh()) # MMS Method of Manufactured Solution
u_exact = 2*sin(pi*x)*sin(pi*y)
G = grad(u_exact)
n = FacetNormal(mesh)
g = dot(G,n)
f = -div(grad(u_exact)) + reaction(u_exact)

labels = boundary_labels(mesh)
ds_neumann = Measure("ds", domain=mesh, subdomain_id=labels['neumannbc'])
ds_dirichlet = Measure("ds", domain=mesh, subdomain_id=labels['dirichletbc'])

F = inner(grad(u), grad(v))*dx + reaction(u)*v*dx- inner(f, v)*dx - g*v*ds_neumann
bcs = [DirichletBC(V, u_exact, labels['dirichletbc'])]

J = u * dx_goal

# Define solver parameters ---------------------
solver_parameters = {
    "max_iterations": 20,
    "output_dir": "output/conv-diff-new",
    #"uniform_refinement": True
    #"use_adjoint_residual": True
}

# # Define actual problem -----------------------
# n = FacetNormal(mesh)
# V = FunctionSpace(mesh, "CG", degree) # Template function space used to define the PDE
# print("DOF = ", V.dim())
# u = Function(V, name="Solution")
# v = TestFunction(V)
# (x, y) = SpatialCoordinate(u.function_space().mesh()) # MMS Method of Manufactured Solution
# u_exact = x*y**2
# f = -div(grad(u_exact))

# F = inner(grad(u), grad(v))*dx - inner(f, v)*dx
# bcs = [DirichletBC(V, u_exact, "on_boundary")]

# J = u * dx
tolerance = 0.000001

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

problem = NonlinearVariationalProblem(F, u, bcs)
GoalAdaptiveNonlinearVariationalSolver(problem, J, tolerance, solver_parameters, exact_solution=u_exact, dual_solver_parameters=sp_dual).solve()


