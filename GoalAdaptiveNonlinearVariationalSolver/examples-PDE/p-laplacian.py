from firedrake import *
from netgen.occ import *
import sys
from goal_adaptivity import GoalAdaptiveNonlinearVariationalSolver
from getlabels import getlabels

def boundary_labels(mesh):
    ngmesh = mesh.netgen_mesh
    names = ngmesh.GetRegionNames(codim=1)
    names_to_labels = {}
    for l in names:
        names_to_labels[l] = tuple(i+1 for i, name in enumerate(names) if name == l)
    return names_to_labels

H = 0.0000005 # Slot width
# --- two half-squares that touch at x = 0 ---
square = WorkPlane().MoveTo(-1, -1).Rectangle(2, 2).Face()
line = WorkPlane().MoveTo(-H/2, -1).Rectangle(H,1).Face()

# Keep coincident edges distinct (no fuse), so we can glue only what we want:
shape = square - line

# Tag only the interface edges; glue the top, leave the bottom open
tol = 0.00000000001
for e in shape.edges:
    mp = e.center
    if abs(mp.x - H/2) < tol: 
        e.name = "neumann"
    else:
        e.name = "dirichlet"

# Mesh and glue only the top interface
geo = OCCGeometry(shape, dim=2)
mesh = Mesh(geo.GenerateMesh(maxh=0.2))

for f in shape.edges: # Assign face labels
    print(f.name)

VTKFile("p-laplcian_mesh.pvd").write(mesh)

# Define solver parameters ---------------------
solver_parameters = {
    "max_iterations": 30,
    "output_dir": "output_final",
    "run_name": "p-laplacian-final-dual",
    "use_adjoint_residual": True,
    "dual_low_method": "solve",
    "dorfler_alpha": 0.3
    #"uniform_refinement": True
    #"use_adjoint_residual": True
}

p   = Constant(4.0)        # e.g. p = 4 (choose > 2 for strong nonlinearity)
eps = Constant(1.0e-10)       # small regularization to avoid singularity at |∇u|=0
degree = 1
V = FunctionSpace(mesh, "CG", degree) # Template function space used to define the PDE
u = Function(V, name="Solution")
v = TestFunction(V)
(x, y) = SpatialCoordinate(mesh) # MMS Method of Manufactured Solution
n = FacetNormal(mesh)
a = (inner(grad(u), grad(u)) + eps**2)**((p - 2)/2)
F = inner(a*grad(u), grad(v))*dx - v*dx   # solve F == 0

labels = boundary_labels(mesh)
ds_dirichlet = Measure("ds", domain=mesh, subdomain_id=labels['dirichlet'])

bcs = [DirichletBC(V, 0.0, labels['dirichlet'])]

J = u * dx

Ju = 0.71755
tolerance = 0.00001

sp_primal = {"snes_monitor": None,
             "snes_linesearch_monitor": None,
             "snes_linesearch_type": "l2"}

for P in [2.0, 2.5, 3.0, 3.5, 4.0]:
    p.assign(P)
    solve(F == 0, u, bcs=bcs, solver_parameters=sp_primal)

print(p)

problem = NonlinearVariationalProblem(F, u, bcs)
GoalAdaptiveNonlinearVariationalSolver(problem, J, tolerance, solver_parameters, exact_goal=Ju, primal_solver_parameters=sp_primal).solve()
