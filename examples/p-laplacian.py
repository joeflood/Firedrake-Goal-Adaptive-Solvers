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
initial_mesh_size = 0.2

outer = WorkPlane().MoveTo(-1, -1).Rectangle(2, 2).Face()
roi   = WorkPlane().MoveTo(-0.5, -0.5).Rectangle(1, 1).Face()
roi.name = "roi"
ring = outer - roi
ring.name = "bulk"
shape = Glue([ring, roi])

tol = 0.00001
for f in shape.edges: # Assign face labels
    if abs(f.center.x - 1) < tol:
        f.name = 'dirichletbc'
    else:
        f.name = 'neumannbc'

geo = OCCGeometry(shape, dim=2)
ngmesh = geo.GenerateMesh(maxh=initial_mesh_size)
mesh = Mesh(ngmesh)

p   = Constant(2.1)        # e.g. p = 4 (choose > 2 for strong nonlinearity)
eps = Constant(1e-2)       # small regularization to avoid singularity at |∇u|=0
lam = Constant(3.0)   # will homotopy 0 → target (e.g. 10)
def reaction(w): return lam*exp(w)

degree = 1
V = FunctionSpace(mesh, "CG", degree) # Template function space used to define the PDE
u = Function(V, name="Solution")
v = TestFunction(V)
(x, y) = SpatialCoordinate(u.function_space().mesh()) # MMS Method of Manufactured Solution
u_exact = x*y**2
G = grad(u_exact)
n = FacetNormal(mesh)

# --- NEW: nonlinear diffusivity a(|∇u|) ---
def a(q):        # q is a vector
    return (eps + inner(q, q))**((p-2)/2)
g = dot( a(G)*G , n)                                  # natural (Neumann) flux n·(a∇u)
f = -div( a(G)*G ) + reaction(u_exact)                # RHS so that u_exact is the solution

labels = boundary_labels(mesh)
ds_neumann = Measure("ds", domain=mesh, subdomain_id=labels['neumannbc'])
ds_dirichlet = Measure("ds", domain=mesh, subdomain_id=labels['dirichletbc'])

F = inner( a(grad(u))*grad(u), grad(v) )*dx + reaction(u)*v*dx- inner(f, v)*dx - g*v*ds_neumann
bcs = [DirichletBC(V, u_exact, labels['dirichletbc'])]

J = u * ds_neumann

# Define solver parameters ---------------------
solver_parameters = {
    "max_iterations": 30,
    "output_dir": "output",
    #"uniform_refinement": True
    #"use_adjoint_residual": True
}

tolerance = 0.000001

problem = NonlinearVariationalProblem(F, u, bcs)
GoalAdaptiveNonlinearVariationalSolver(problem, J, tolerance, solver_parameters, u_exact).solve()


