from firedrake import *
from netgen.occ import *
import sys
from goaladadaptiveeigensolver import GoalAdaptiveEigenSolver

# User knobs
nx = 20
degree = 1
NEV_SOLVE = 15

# Mesh and spaces
mesh = Mesh(unit_square.GenerateMesh(maxh=1/nx))

V  = FunctionSpace(mesh, "CG", degree)
u  = TrialFunction(V); v = TestFunction(V)
A  = inner(grad(u), grad(v)) * dx
M  = inner(u, v) * dx
bcs = [DirichletBC(V, 0.0, "on_boundary")]

# Pick a target by (m,n) OR set 'target' directly to a float
m_t, n_t = 3, 3
target = float(np.pi**2 * (m_t*m_t + n_t*n_t))  # override this with a number if you like
print("Target eigenvaue: ", target)
tolerance = 0.001

solver_parameters = {
    "max_iterations": 10,
    "output_dir": "output/eigen_problem",
    "manual_indicators": False,
    "dual_extra_degree": 1,
    "use_adjoint_residual": True,
    "primal_low_method": "interpolate",
    "dual_low_method": "interpolate",
    "write_mesh": "no",
    "write_solution": "no"
    #"uniform_refinement": True
    #"use_adjoint_residual": True
}

problem = LinearEigenproblem(A,M,bcs)
solver = GoalAdaptiveEigenSolver(problem, target, tolerance, solver_parameters=solver_parameters, exact_solution=target)
solver.solve()