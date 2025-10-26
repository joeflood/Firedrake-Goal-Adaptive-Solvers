from firedrake import *
from netgen.occ import *
import sys
from goaladadaptiveeigensolver_complex import GoalAdaptiveEigenSolverComplex

# ----- OCC mesh (2D rectangle) -----
initial_mesh_size = 0.2

box2 = WorkPlane().MoveTo(0, 0).Rectangle(1, 1).Face()
shape = box2
geo = OCCGeometry(shape, dim = 2)
ngmesh = geo.GenerateMesh(maxh=initial_mesh_size)
mesh = Mesh(ngmesh)
                          # Firedrake mesh from Netgen OCC

# ----- Vector H1 space and trial/test -----
V = VectorFunctionSpace(mesh, "CG", 1)
u = TrialFunction(V)
v = TestFunction(V)

# ----- Symmetric part (vector Laplacian) + mass -----
K = inner(grad(u), grad(v)) * dx
M = inner(u, v) * dx

# ----- Skew-symmetric 0th-order “rotation” term:  (J u)·v  -----
# J = [[0, -1], [1, 0]] rotates by +90°, hence skew in L2
J = as_tensor(((0.0, -1.0),
               (1.0,  0.0)))
Ju = dot(as_tensor(((0.0, -1.0),
               (1.0,  0.0))), u)                               # vector
omega = Constant(6.0)                        # strength; increase if needed
G = Constant(6.0) * inner(dot(as_tensor(((0.0, -1.0),
               (1.0,  0.0))), u)  , v) * dx                # G(v,u) = -G(u,v)

A = inner(grad(u), grad(v)) * dx + Constant(6.0) * inner(dot(as_tensor(((0.0, -1.0),
               (1.0,  0.0))), u)  , v) * dx                                    # non-self-adjoint (real)

# Dirichlet BCs just to remove rigid/constant modes (keeps it simple)
bcs = [DirichletBC(V, Constant((0.0, 0.0)), "on_boundary")]


target = 2.0 * pi**2
tolerance = 0.001


problem = LinearEigenproblem(A,M,bcs)
solver = GoalAdaptiveEigenSolverComplex(problem, target, tolerance, solver_parameters={}, exact_solution=target+6j)
solver.solve()