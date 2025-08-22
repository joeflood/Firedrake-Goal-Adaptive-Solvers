from firedrake import *
from netgen.occ import *
import os, time, itertools, gc
import sys
sys.path.insert(0, "./algorithm")

from goal_adaptivity import GoalAdaptiveNonlinearVariationalSolver

# --- one run ------------------------------------------------------------
def run_case(primal_low_method, dual_low_method, out_root="output/conv-diff-batch", initial_mesh_size=0.1):
    # unique folder per combo (plus a common timestamp suffix so repeated batches don't overwrite)
    outdir = os.path.join(out_root, f"{primal_low_method}-{dual_low_method}")
    os.makedirs(outdir, exist_ok=True)

    # --- geometry / fresh mesh each run
    shape = WorkPlane().MoveTo(-1, -1).Rectangle(2, 2).Face()
    tol_edge = 1e-12
    for e in shape.edges:
        if abs(e.center.x - 1) < tol_edge:   e.name = "x1"
        elif abs(e.center.x + 1) < tol_edge: e.name = "xm1"
        elif abs(e.center.y - 1) < tol_edge: e.name = "y1"
        elif abs(e.center.y + 1) < tol_edge: e.name = "ym1"
    geo    = OCCGeometry(shape, dim=2)
    ngmesh = geo.GenerateMesh(maxh=initial_mesh_size)
    mesh   = Mesh(ngmesh)

    # --- FE spaces and problem data
    degree = 1
    V  = FunctionSpace(mesh, "CG", degree)
    u  = Function(V, name="u")
    v  = TestFunction(V)

    eps = Constant(1.0/200.0)
    w   = Constant(as_vector([0.0, 1.0]))
    F   = eps*inner(grad(u), grad(v))*dx + inner(w, grad(u))*v*dx

    x, y = SpatialCoordinate(mesh)
    exact_sol = x*(1 - exp((y - 1.0)/eps)) / (1 - exp(-2.0/eps))
    bcs = DirichletBC(V, exact_sol, "on_boundary")

    # goal functional (example)
    M = inner(u, u)*dx
    tol = 1e-9

    solver_parameters = {
        "max_iterations": 10,
        "output_dir": outdir,                 # <-- unique per run
        "manual_indicators": False,
        "dual_extra_degree": 1,
        "use_adjoint_residual": True,
        "primal_low_method": primal_low_method,   # "solve" | "project" | "interpolate"
        "dual_low_method": dual_low_method,       # "solve" | "project" | "interpolate"
        "write_mesh": "no",
        "write_solution": "no"
    }

    problem = NonlinearVariationalProblem(F, u, bcs)
    adapt = GoalAdaptiveNonlinearVariationalSolver(
        problem, M, tol, solver_parameters, exact_solution=exact_sol
    )
    print(f"\n=== RUN {primal_low_method=}, {dual_low_method=} → {outdir} ===")
    adapt.solve()
    # free memory between runs
    del adapt, problem, V, u, v, mesh, ngmesh, shape
    gc.collect()
    return outdir

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
