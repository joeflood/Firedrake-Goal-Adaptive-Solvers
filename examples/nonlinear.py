from firedrake import *
from netgen.occ import *
import sys
from goal_adaptivity import GoalAdaptiveNonlinearVariationalSolver

initial_mesh_size = 0.2
# Initial mesh

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
    if abs(f.center.x - 1) < tol or abs(f.center.y - 1) < tol:
        f.name = 'dirichletbcs'



# box1 = Box(Pnt(-1,0,-1), Pnt(0,1,0))
# box2 = Box(Pnt(0,0,-1), Pnt(1,1,0))
# box3 = Box(Pnt(0,-1,-1), Pnt(1,0,0))
# shape = box1 + box2 + box3

# for f in shape.faces: # Assign face labels
#     if f.center.x == -1:
#         f.name = "goal_face"
#     if f.center.x == 1 or f.center.y == 1:
#         f.name = "dirichletbcs"

geo = OCCGeometry(shape, dim=2)
ngmesh = geo.GenerateMesh(maxh=initial_mesh_size)
mesh = Mesh(ngmesh)

def boundary_labels(mesh):
    ngmesh = mesh.netgen_mesh
    names = ngmesh.GetRegionNames(codim=1)
    names_to_labels = {}
    for l in names:
        names_to_labels[l] = tuple(i+1 for i, name in enumerate(names) if name == l)
    return names_to_labels

degree = 1
V = FunctionSpace(mesh, "CG", degree) # Template function space used to define the PDE
u = Function(V, name="Solution")
v = TestFunction(V)
(x, y) = SpatialCoordinate(u.function_space().mesh()) # MMS Method of Manufactured Solution
u_exact = (x-1)*(y-1)**2
G = grad(u_exact)
n = FacetNormal(mesh)
g = dot(G,n)
f = -div(grad(u_exact)) + u_exact**3

labels = boundary_labels(mesh)
ds_goal = Measure("ds", domain=mesh, subdomain_id=labels['goal_face'])

F = inner(grad(u), grad(v))*dx + u**3*v*dx- inner(f, v)*dx - g*v*ds
bcs = [DirichletBC(V, u_exact, labels['dirichletbcs'])]

J = dot(grad(u), n)*ds_goal

# Define solver parameters ---------------------
solver_parameters = {
    "degree": 1,
    "dual_solve_method": "high_order",
    "dual_solve_degree": "degree + 1",
    "residual_solve_method": "automatic",
    "residual_degree": "degree",
    "dorfler_alpha": 0.5,
    "goal_tolerance": 0.000001,
    "max_iterations": 20,
    "output_dir": "output/nonlinear",
    "write_at_iteration": True,
    #"residual": "both"
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

problem = NonlinearVariationalProblem(F, u, bcs)
GoalAdaptiveNonlinearVariationalSolver(problem, J, tolerance, solver_parameters, u_exact).solve()


