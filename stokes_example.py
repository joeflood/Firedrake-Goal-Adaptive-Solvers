from firedrake import *
from netgen.occ import *
import sys
from goaladadaptiveeigensolver_complex import GoalAdaptiveEigenSolverComplex



# -------- OCC mesh: unit square --------
initial_mesh_size = 0.2
rect = WorkPlane().MoveTo(0, 0).Rectangle(1.0, 1.0).Face()
geo  = OCCGeometry(rect, dim=2)
ngm  = geo.GenerateMesh(maxh=initial_mesh_size)
mesh = Mesh(ngm)

# -------- Taylor–Hood spaces --------
k = 2
V = VectorFunctionSpace(mesh, "CG", k)     # velocity (P2)
Q = FunctionSpace(mesh, "CG", k-1)         # pressure (P1)
W = V * Q

t = TrialFunction(W)
test = TestFunction(W)

u, p = split(t)
v, q = split(test)

# Bilinear (Stokes operator)
A = ( inner(grad(u), grad(v)) * dx
    - div(v)*p * dx
    + q*div(u) * dx )

# Mass only on velocity (pressure part = 0)
delta_p = Constant(1e-10)         # 1e-10..1e-6 is typical
M = inner(u, v) * dx + delta_p * p * q * dx

# No-slip
bcs = [DirichletBC(W.sub(0), Constant((0.0, 0.0)), "on_boundary")]

# (Optional) if your SLEPc build dislikes singular M, add a tiny pressure mass:
# delta = Constant(1e-12)
# M += delta * p*q * dx

# -------- Solve a few smallest eigenpairs --------
target = 52.3446911  # first Stokes/buckling eigenvalue on unit square (reference) EXACT
#target = 92

nev = 5
tolerance = 0.00001


problem = LinearEigenproblem(A,M,bcs)
solver = GoalAdaptiveEigenSolverComplex(problem, target, tolerance, solver_parameters={"self_adjoint": True}, exact_solution=target)
solver.solve()