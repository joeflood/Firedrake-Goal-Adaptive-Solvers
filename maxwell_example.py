from firedrake import *
from netgen.occ import *
import sys
from goaladadaptiveeigensolver import GoalAdaptiveEigenSolver
from ufl import conj

from petsc4py import PETSc
is_complex = PETSc.ScalarType is complex
print("ScalarType:", PETSc.ScalarType)

# User knobs
#nx = 20
degree = 1
NEV_SOLVE = 15

# Mesh and spaces
#mesh = Mesh(unit_square.GenerateMesh(maxh=1/nx))

# Define initial mesh ---------------------
initial_mesh_size = 0.2

box1 = WorkPlane().MoveTo(-1, 0).Rectangle(1, 1).Face()
box2 = WorkPlane().MoveTo(0, 0).Rectangle(1, 1).Face()
box3 = WorkPlane().MoveTo(0, -1).Rectangle(1, 1).Face()
shape = box1 + box2 + box3
geo = OCCGeometry(shape, dim = 2)
ngmesh = geo.GenerateMesh(maxh=initial_mesh_size)
mesh = Mesh(ngmesh)

V  = FunctionSpace(mesh, "N1curl", degree)
u  = TrialFunction(V); v = TestFunction(V)
A  = inner(curl(conj(v)), curl(u)) * dx
M  = inner(conj(v), u) * dx
bcs = [DirichletBC(V, Constant((0.0, 0.0)), "on_boundary")]

# Pick a target by (m,n) OR set 'target' directly to a float
m_t, n_t = 1, 1
target = 12.5723873200  # override this with a number if you like
print("Target eigenvaue: ", target)
tolerance = 0.001

solver_parameters = {
    "max_iterations": 10,
    "output_dir": "output/maxwell_eig6",
    "manual_indicators": False,
    "dual_extra_degree": 1,
    "use_adjoint_residual": True,
    "primal_low_method": "interpolate",
    "dual_low_method": "interpolate",
    "uniform_refinement": True
    #"use_adjoint_residual": True
}

problem = LinearEigenproblem(A,M,bcs)
solver = GoalAdaptiveEigenSolver(problem, target, tolerance, solver_parameters=solver_parameters, exact_solution=target)
solver.solve()