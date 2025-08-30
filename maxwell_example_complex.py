from firedrake import *
from netgen.occ import *
import sys
from goaladadaptiveeigensolver_complex import GoalAdaptiveEigenSolverComplex

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
A  = inner(curl(u), curl(v)) * dx
M  = inner(u, v) * dx
bcs = [DirichletBC(V, Constant((0.0, 0.0)), "on_boundary")]

#target = 12.5723873200 
target = 23.344371957137 # 9th eigenvalue
#target = 21.4247335393 # 8th eigenvalue
#target = 1.47562182397 # 1st eigenvalue
target = pi**2 # 3rd and 4th eigenvalue

print("Target eigenvaue: ", target)
tolerance = 0.00001

solver_parameters = {
    "max_iterations": 30,
    "output_dir": "output_eigenproblems/maxwell/eig3and4_mag",
    "dual_extra_degree": 1,
    "self_adjoint": True,
    "dorfler_alpha": 0.5
    #"uniform_refinement": True
    #"use_adjoint_residual": True
}

problem = LinearEigenproblem(A,M,bcs)
solver = GoalAdaptiveEigenSolverComplex(problem, target, tolerance, solver_parameters=solver_parameters, exact_solution=target)
solver.solve()