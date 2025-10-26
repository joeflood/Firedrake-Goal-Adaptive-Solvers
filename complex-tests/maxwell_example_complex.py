from firedrake import *
from netgen.occ import *
import sys
from goaladadaptiveeigensolver_complex import GoalAdaptiveEigenSolverComplex

# User knobs
#nx = 20
degree = 3
NEV_SOLVE = 5

# Mesh and spaces
#mesh = Mesh(unit_square.GenerateMesh(maxh=1/nx))

# Define initial mesh ---------------------
initial_mesh_size = 0.5

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
#target = 23.344371957137 # 9th eigenvalue
#target = 21.4247335393 # 8th eigenvalue
#target = 1.47562182397 # 1st eigenvalue
target = pi**2 # 3rd and 4th eigenvalue

print("Target eigenvaue: ", target)
tolerance = 0.00000000000001

adaptive_parameters = {
    "max_iterations": 30,
    "run_name": "eig3_naive_uniform",
    "output_dir": "output_eigenproblems/maxwell_new",
    "dual_extra_degree": 1,
    "self_adjoint": True,
    "dorfler_alpha": 0.5,
    "write_mesh": "none",
    "NEV_SOLVE": NEV_SOLVE,
    "uniform_refinement": True
    #"use_adjoint_residual": True
}

# Attempts at stopping the krylov space from collapsing at iteration 22
solver_parameters1 = {
  "st_type": "sinvert",
  "st_ksp_type": "gmres",
  "st_pc_type": "hypre",
  "st_ksp_rtol": 1e-10,
  "st_ksp_max_it": 500
}
solver_parameters2 = {
  "st_type": "sinvert",
  "st_ksp_type": "preonly",
  "st_pc_type": "lu",
  "st_pc_factor_mat_solver_type": "mumps"
}

problem = LinearEigenproblem(A,M,bcs)
solver = GoalAdaptiveEigenSolverComplex(problem, target, tolerance, adaptive_parameters=adaptive_parameters, 
                                        solver_parameters=solver_parameters2, exact_solution=target)
solver.solve()