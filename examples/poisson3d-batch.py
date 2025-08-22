from firedrake import *
from netgen.occ import *
import os, time, itertools, gc
import sys
sys.path.insert(0, "./algorithm")

from goal_adaptivity import GoalAdaptiveNonlinearVariationalSolver
from goal_adaptivity import getlabels

# --- one run ------------------------------------------------------------
def run_case(primal_low_method, dual_low_method, out_root="output/poisson3d-batch", initial_mesh_size=0.1):
    # unique folder per combo (plus a common timestamp suffix so repeated batches don't overwrite 

    # Initial mesh
    box1 = Box(Pnt(-1,0,-1), Pnt(0,1,0))
    box2 = Box(Pnt(0,0,-1), Pnt(1,1,0))
    box3 = Box(Pnt(0,-1,-1), Pnt(1,0,0))
    shape = box1 + box2 + box3

    tol = 0.00000001
    for f in shape.faces: # Assign face labels
        if abs(f.center.x + 1) < tol:
            f.name = "goal_face"
        elif abs(f.center.x - 1) < tol or abs(f.center.y - 1) < tol:
            f.name = "dirichletbcs"
        else: 
            f.name = "neumannbcs"  

    geo = OCCGeometry(shape)
    ngmesh = geo.GenerateMesh(maxh=initial_mesh_size)
    mesh = Mesh(ngmesh)

    degree = 1
    n = FacetNormal(mesh)
    V = FunctionSpace(mesh, "CG", degree, variant="integral") # Template function space used to define the PDE
    u = Function(V, name="Solution")
    v = TestFunction(V)
    (x, y, z) = SpatialCoordinate(u.function_space().mesh()) # MMS Method of Manufactured Solution
    u_exact = (x-1)*(y-1)**2
    G = as_vector(((y-1)**2, 2*(x-1)*(y-1), 0.0))
    g = dot(G,n)
    f = -div(grad(u_exact))

    labels = getlabels(mesh, 1)
    ds_goal = Measure("ds", domain=mesh, subdomain_id=labels['goal_face'])
    dxm     = Measure("dx", domain=mesh)
    ds_neumann     = Measure("ds", domain=mesh, subdomain_id=labels['neumannbcs']+labels['goal_face'])
    ds_dirichlet = Measure("ds", domain=mesh, subdomain_id=labels['dirichletbcs'])

    F = inner(grad(u), grad(v))*dxm - inner(f, v)*dxm - g*v*ds_neumann
    bcs = [DirichletBC(V, u_exact, labels['dirichletbcs'])]

    J = u*ds_goal
    tolerance = 0.00001
    solver_parameters = {
        "max_iterations": 5,
        "output_dir": "./poisson3d-batch",                 # <-- unique per run
        "manual_indicators": False,
        "dual_extra_degree": 1,
        "use_adjoint_residual": True,
        "primal_low_method": primal_low_method,   # "solve" | "project" | "interpolate"
        "dual_low_method": dual_low_method,       # "solve" | "project" | "interpolate"
        "write_mesh": "no",
        "write_solution": "no",
        "results_file_name": f"{primal_low_method}-{dual_low_method}"
    }

    problem = NonlinearVariationalProblem(F, u, bcs)
    adapt = GoalAdaptiveNonlinearVariationalSolver(
        problem, J, tolerance, solver_parameters, exact_solution=u_exact
    )
    print(f"\n=== RUN {primal_low_method=}, {dual_low_method=} ===")
    adapt.solve()
    # free memory between runs
    del adapt, problem, V, u, v, mesh, ngmesh, shape
    gc.collect()
    return

# --- batch over all combinations ----------------------------------------
if __name__ == "__main__":
    methods = ("solve", "project", "interpolate")
    for plm, dlm in itertools.product(methods, methods):
        try:
            run_case(plm, dlm)
        except Exception as e:
            # don't let one failure kill the batch
            import traceback
            print(f"[ERROR] {plm=}, {dlm=}: {e}")
            traceback.print_exc()
