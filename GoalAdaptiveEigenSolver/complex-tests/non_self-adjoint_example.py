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
V = FunctionSpace(mesh, "CG", 1)
u, v = TrialFunction(V), TestFunction(V)

alpha = Constant(0.0)         # real part (damping/shift if wanted)
beta  = Constant(2.0)         # imaginary impedance  => non-self-adjoint
A = inner(grad(u), grad(v))*dx + (alpha + 1j*beta)*inner(u, v)*ds
M = inner(u, v)*dx
bcs = []                      # pure Robin; or mix with Dirichlet on parts

# (Optional) if your SLEPc build dislikes singular M, add a tiny pressure mass:
# delta = Constant(1e-12)
# M += delta * p*q * dx

# -------- Solve a few smallest eigenpairs --------
target = 2  # first Stokes/buckling eigenvalue on unit square (reference) EXACT
#target = 92

nev = 5
tolerance = 0.00001


problem = LinearEigenproblem(A,M,bcs)
solver = GoalAdaptiveEigenSolverComplex(problem, target, tolerance, solver_parameters={"self_adjoint": True}, exact_solution=target)
solver.solve()