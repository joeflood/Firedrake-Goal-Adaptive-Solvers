from firedrake import *
from netgen.occ import *
import sys
from goal_adaptivity import GoalAdaptiveNonlinearVariationalSolver
from goal_adaptivity import getlabels

def boundary_labels(mesh):
    ngmesh = mesh.netgen_mesh
    names = ngmesh.GetRegionNames(codim=1)
    names_to_labels = {}
    for l in names:
        names_to_labels[l] = tuple(i+1 for i, name in enumerate(names) if name == l)
    return names_to_labels

sys.path.insert(0, "./algorithm")
initial_mesh_size = 0.1
shape = WorkPlane().MoveTo(-1, -1).Rectangle(2, 2).Face()

tol = 0.00001
for f in shape.edges: # Assign face labels
    if abs(f.center.x -1) < tol:
        f.name = 'x1'
        print("x=1 named")
    elif abs(f.center.x + 1) < tol:
        f.name = 'xm1'
        print("x=-1 named")
    elif abs(f.center.y - 1) < tol:
        f.name = 'y1'
        print("y=1 named")
    elif abs(f.center.y + 1) < tol:
        f.name = "ym1"
        print("y=-1 named")
    else:
        print("Unrecognised edge")

geo = OCCGeometry(shape, dim=2)
ngmesh = geo.GenerateMesh(maxh=initial_mesh_size)
mesh = Mesh(ngmesh)


labels = boundary_labels(mesh)

# Define solver parameters ---------------------
solver_parameters = {
    "max_iterations": 20,
    "output_dir": "output/conv-diff-new",
    "manual_indicators": False,
    "dual_extra_degree": 3,
    "use_adjoint_residual": True,
    "primal_low_method": "project",
    "dual_low_method": "project",
    "write_mesh": "no",
    "write_solution": "no"
    #"uniform_refinement": True
    #"use_adjoint_residual": True
}

degree = 1
V = FunctionSpace(mesh,"CG",degree)
u = Function(V)
v = TestFunction(V)
eps = Constant(1/200)
vel = Constant(as_vector([0, 1]))
F = eps * inner(grad(u),grad(v)) *dx + inner(vel, grad(u)) * v * dx 


(x,y) = SpatialCoordinate(mesh)
# bc_ym1 = DirichletBC(V, x, labels['ym1'])
# bc_y1 = DirichletBC(V, 0.0, labels['y1'])
# bc_xm1 = DirichletBC(V, 0.0, labels['xm1'])
# bc_x1  = DirichletBC(V, conditional(le(abs(y - 1.0), tol), 0.0, 2.0), labels['x1'])

# bcs = [bc_ym1, bc_y1, bc_xm1, bc_x1]

M = 0.5*eps * inner(u,u) *dx  
tolerance = 0.000000001

exact_sol = x * (1-exp((y-1)/eps)) / (1-exp(-2/eps))
bcs = DirichletBC(V, exact_sol, "on_boundary")

problem = NonlinearVariationalProblem(F, u, bcs)

# Manual indicator test for comparison
class GoalAdaptiveConvDiff(GoalAdaptiveNonlinearVariationalSolver):
    def manual_error_indicators(self):
        print("[MANUAL] Computing local refinement indicators (η_K)...")
        s = self.solverctx
        n = FacetNormal(self.mesh)
        DG0 = FunctionSpace(self.mesh, "DG", degree=0)
        test = TestFunction(DG0)

        eps = Constant(1/200)
        vel = Constant(as_vector([0, 1]))   
        self.etaT = assemble(
            eps * dot(grad(self.u),grad(self.z_err)) * test * dx
            + dot(vel , grad(self.u)) * self.z_err * test * dx
            )
        


#adaptive_problem = GoalAdaptiveConvDiff(problem, M, tolerance, solver_parameters, exact_solution=exact_sol)
#adaptive_problem.solve()


adaptive_problem = GoalAdaptiveNonlinearVariationalSolver(problem, M, tolerance, solver_parameters, exact_solution=exact_sol)
adaptive_problem.solve()


