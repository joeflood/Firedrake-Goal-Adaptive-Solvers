# --- put this at the very top (before importing firedrake) to kill the OMP warning ---
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

from firedrake import *
from firedrake.eigensolver import LinearEigenproblem, LinearEigensolver
from netgen.occ import *
import numpy as np
from firedrake.__future__ import interpolate as interp  # symbolic interpolate

# =======================
# User knobs
# =======================
nx = 50
degree = 1

# Pick a target by (m,n) OR set 'target' directly to a float
m_t, n_t = 5, 6
target = float(np.pi**2 * (m_t*m_t + n_t*n_t))  # override this with a number if you like
N_SHOW = 10                                     # number of closest eigenpairs to list

# Extra compute slack (compute more than you show, to be safe around degeneracies)
NEV_SOLVE = 100

# =======================
# Mesh and spaces
# =======================
mesh = Mesh(unit_square.GenerateMesh(maxh=1/nx))

V  = FunctionSpace(mesh, "CG", degree)
u  = TrialFunction(V); v = TestFunction(V)
A  = inner(grad(u), grad(v)) * dx
M  = inner(u, v) * dx
bcs = [DirichletBC(V, 0.0, "on_boundary")]

Vp1 = FunctionSpace(mesh, "CG", degree + 1)
up1 = TrialFunction(Vp1); vp1 = TestFunction(Vp1)
Ap1 = inner(grad(up1), grad(vp1))*dx
Mp1 = inner(up1, vp1)*dx
bcs_p1 = [DirichletBC(Vp1, 0.0, "on_boundary")]

# rich space for exact modes & errors
V_exact = FunctionSpace(mesh, "CG", degree + 8)

# exact eigenvalue list for printing/identification
mmax, nmax = 20, 20
exact_list = [(np.pi**2 * (m*m + n*n), m, n)
              for m in range(1, mmax+1) for n in range(1, nmax+1)]
exact_list.sort(key=lambda t: t[0])

x, y = SpatialCoordinate(mesh)
def exact_expr(m, n):
    return sin(m*pi*x) * sin(n*pi*y)

def l2_normalize(f):
    nrm = assemble(inner(f, f)*dx)**0.5
    if nrm > 0:
        f.assign(f/nrm)
    return f

# ------------------------------
# Eigensolve helpers
# ------------------------------
def solve_eigs(Aform, Mform, Vspace, bcs, nev, solver_parameters):
    prob = LinearEigenproblem(Aform, Mform, bcs=bcs, restrict=True)
    es = LinearEigensolver(prob, n_evals=nev, solver_parameters=solver_parameters)
    nconv = es.solve()
    lam, vecs = [], []
    for i in range(min(nconv, nev)):
        lam.append(es.eigenvalue(i))
        vr, vi = es.eigenfunction(i)
        vecs.append(l2_normalize(vr))
    return lam, vecs

def sort_by_target(lams, vecs, tgt):
    order = sorted(range(len(lams)), key=lambda i: abs(lams[i] - tgt))
    return [lams[i] for i in order], [vecs[i] for i in order]

def best_match(target, pool, used):
    """Return (index, copy_aligned) maximizing |(target, v_i)| in L2, skipping 'used'."""
    best_i, best_c = -1, 0.0
    for i, w in enumerate(pool):
        if i in used: continue
        c = float(assemble(inner(target, w) * dx))
        if abs(c) > abs(best_c):
            best_i, best_c = i, c
    if best_i < 0:
        raise RuntimeError("No available vector to match.")
    w = pool[best_i].copy(deepcopy=True)
    if best_c < 0: w.assign(-w)
    used.add(best_i)
    return best_i, w

# shift-and-invert centered at target ⇒ nearest eigenpairs are "largest magnitude"
solver_parameters_target = {
    "eps_gen_hermitian": None,
    "eps_target": target
}

def lambda_max_discrete():
    params = {
        "eps_gen_hermitian": None,
    }
    lam, _ = solve_eigs(A, M, V, bcs, nev=2, solver_parameters=params)
    return lam

print("λ_max(discrete) ≈", lambda_max_discrete()) # Doesn't work yet.

# --- solve once per space (compute > show) ---
lam_h,  V_vecs   = solve_eigs(A,  M,  V,  bcs,    nev=NEV_SOLVE, solver_parameters=solver_parameters_target)
lam_hp1, Vp1_vecs = solve_eigs(Ap1, Mp1, Vp1, bcs_p1, nev=NEV_SOLVE, solver_parameters=solver_parameters_target)

# Sort by closeness to the target
lam_h,  V_vecs   = sort_by_target(lam_h,  V_vecs,  target)
lam_hp1, Vp1_vecs = sort_by_target(lam_hp1, Vp1_vecs, target)



# ------------------------------
# Residual-based indicators
# ------------------------------
def both(u):
    return u("+") + u("-")

def residual(form, test):  # Residual helper
    vtrial = form.arguments()[0]
    return replace(form, {vtrial: test})

sp_cell2   = {"mat_type": "matfree", "snes_type": "ksponly", "ksp_type": "cg", "pc_type": "jacobi", "pc_hypre_type": "pilut"}
sp_facet1  = {"mat_type": "matfree", "snes_type": "ksponly", "ksp_type": "cg", "pc_type": "jacobi", "pc_hypre_type": "pilut"}

def automatic_error_indicators(z_err, F):
    # cell residual
    dim = mesh.topological_dimension()
    cell = mesh.ufl_cell()
    variant = "integral"

    B  = FunctionSpace(mesh, "B", dim+1, variant=variant)
    bubbles = Function(B).assign(1)

    DG = FunctionSpace(mesh, "DG", 1, variant=variant)
    uc = TrialFunction(DG); vc = TestFunction(DG)
    ac = inner(uc, bubbles*vc)*dx
    Lc = residual(F, bubbles*vc)

    Rcell = Function(DG, name="Rcell")
    solve(ac == Lc, Rcell, solver_parameters=sp_cell2)

    # facet residual
    FB = FunctionSpace(mesh, "FB", dim, variant=variant)
    cones = Function(FB).assign(1)

    el = BrokenElement(FiniteElement("FB", cell=cell, degree=1+dim, variant=variant))
    Q  = FunctionSpace(mesh, el)
    Qtest = TestFunction(Q); Qtrial = TrialFunction(Q)
    Lf = residual(F, Qtest) - inner(Rcell, Qtest)*dx
    af = both(inner(Qtrial/cones, Qtest))*dS + inner(Qtrial/cones, Qtest)*ds

    Rhat = Function(Q)
    solve(af == Lf, Rhat, solver_parameters=sp_facet1)
    Rfacet = Rhat/cones

    # indicators
    DG0 = FunctionSpace(mesh, "DG", degree=0)
    test = TestFunction(DG0)
    etaT = assemble(
        inner(inner(Rcell, z_err), test)*dx
      + inner(avg(inner(Rfacet, z_err)), both(test))*dS
      + inner(inner(Rfacet, z_err), test)*ds
    )
    return etaT

# ------------------------------
# Identify the analytic (m,n) matching a function
# ------------------------------
# Prebuild normalized exact modes for matching
exact_funcs = []
for lam_ex, m, n in exact_list:
    f = Function(V_exact); f.interpolate(exact_expr(m, n)); l2_normalize(f)
    exact_funcs.append((lam_ex, m, n, f))

def identify_exact(vfunc):
    best = (None, None, None, 0.0)  # (lam, m, n, |corr|)
    for lam_ex, m, n, f in exact_funcs:
        c = abs(float(assemble(inner(vfunc, f)*dx)))
        if c > best[3]:
            best = (lam_ex, m, n, c)
    return best[0], best[1], best[2]

# ------------------------------
# Targeted run: list nearest computed eigenpairs (with effectivities)
# ------------------------------
def run_targeted(title, n_show):
    print(title)
    print("\n# k  (m,n)  λ_h  sigma_h     error_exact     error_predicted     effectivity1     sum(local errors)     effectivity2")

    rows = []
    used_p1 = set()

    for k in range(1, min(n_show, len(lam_h)) + 1):
        v_h  = V_vecs[k-1]
        lamh = lam_h[k-1]

        # match to a p+1 eigenfunction (reference)
        _, v_used = best_match(v_h, Vp1_vecs, used_p1)

        # identify the analytic mode (m,n) closest to v_used
        lam_exact, m, n = identify_exact(v_used)

        # build phi_h in V (interpolant of v_used)
        phi_h = Function(V)
        phi_h.interpolate(v_used)

        # sigma_h = 1/2 ||v_used - v_h||^2
        e_sigma = v_used - v_h
        sigma_h = 0.5 * assemble(inner(e_sigma, e_sigma) * dx)

        # RHS uses only the out-of-V part of v_used
        e = v_used - phi_h
        rhs = assemble(inner(grad(v_h), grad(e)) * dx) - lamh * assemble(inner(v_h, e) * dx)

        error_exact = abs(lam_exact - lamh)
        denom = 1.0 - sigma_h
        error_pred = abs(rhs / denom) if abs(denom) > 1e-14 else float("nan")
        diff = abs(error_exact - error_pred)
        effectivity = (error_pred / error_exact) if error_exact > 0 else float("nan")
        rows.append((k, m, n, error_exact, error_pred, diff))

        # ------- Local indicators (manual) -------
        n_f  = FacetNormal(mesh)
        DG0  = FunctionSpace(mesh, "DG", degree=0)
        test0 = TestFunction(DG0)
        eta_T = assemble(
              inner(div(grad(v_h)), e * test0) * dx
            + (-lamh * v_h * e) * test0 * dx
            + (jump(grad(v_h), n_f) * e) * 0.5*both(test0) * dS
            + inner(dot(-grad(v_h), n_f), e * test0) * ds
        )
        with eta_T.dat.vec_ro as vvec:
            eta_array = vvec.getArray().copy()
        total = np.abs(eta_array).sum()
        total_normalised = total/denom if abs(denom) > 1e-14 else float("nan")
        eff2 = (total_normalised / error_exact) if error_exact > 0 else float("nan")

        # ------- Local indicators (automatic) -------
        vT = TestFunction(V)
        form = inner(grad(v_h), grad(vT)) * dx - lamh * inner(v_h, vT) * dx
        eta_T_auto = automatic_error_indicators(e, form)
        with eta_T_auto.dat.vec_ro as vvec:
            eta_array_auto = vvec.getArray().copy()
        total_auto = np.abs(eta_array_auto).sum()
        total_normalised_auto = total_auto/denom if abs(denom) > 1e-14 else float("nan")
        eff2_auto = (total_normalised_auto / error_exact) if error_exact > 0 else float("nan")

        # print in your original format
        print(f"{k:2d}  {lamh: .6f}   ({m},{n})   {sigma_h: .6f}   {error_exact: .6e}     {error_pred: .6e}    {effectivity: .3f}     {total_normalised: .6e}      {eff2: .3f} ")
        print(f"Automatic:                                                            {total_normalised_auto: .6e}      {eff2_auto: .3f} ")

    return rows

# =======================
# Run
# =======================
rows_target = run_targeted(f"Nearest to target λ≈{target:.6f} (showing {N_SHOW}):", n_show=N_SHOW)
