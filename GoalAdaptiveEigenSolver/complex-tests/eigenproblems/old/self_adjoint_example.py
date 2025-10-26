from firedrake import *
from firedrake.eigensolver import LinearEigenproblem, LinearEigensolver
from netgen.occ import *
import numpy as np
from firedrake.__future__ import interpolate as interp

# Mesh 
nx = 10
mesh = Mesh(unit_square.GenerateMesh(maxh=1/nx))
# mesh = UnitSquareMesh(nx, nx)

degree = 1
dual_degree = 2
V  = FunctionSpace(mesh, "CG", degree)
u  = TrialFunction(V); v  = TestFunction(V)
A  = inner(grad(u), grad(v)) * dx
M  = inner(u, v) * dx
bcs = [DirichletBC(V, 0.0, "on_boundary")]

Vd = FunctionSpace(mesh, "CG", degree + 8)
ud = TrialFunction(Vd); vd = TestFunction(Vd)
Ad = inner(grad(ud), grad(vd)) * dx
Md = inner(ud, vd) * dx
bcs_d = [DirichletBC(Vd, 0.0, "on_boundary")]

# ----------------------------
# Exact eigenvalues list (yours)
# ----------------------------
mmax, nmax = 6, 6
exact_list = [(np.pi**2 * (m*m + n*n), m, n)
              for m in range(1, mmax+1)
              for n in range(1, nmax+1)]
exact_list.sort(key=lambda t: t[0])
exact_lambdas = np.array([t[0] for t in exact_list])

# ----------------------------
# Helper: exact eigenfunction v_{m,n} and Ih v (phi_h)
# ----------------------------
x, y = SpatialCoordinate(mesh)

def build_exact_v(m, n, space):
    v = Function(space, name=f"v_{m}_{n}")
    v.interpolate(sin(m*pi*x) * sin(n*pi*y))
    nrm = np.sqrt(assemble(inner(v, v) * dx))
    if nrm > 0:
        v.assign(v / nrm)  # L2-normalize
    return v

def build_phi_h(m, n):
    phi = Function(V, name=f"phi_{m}_{n}")
    phi.interpolate(sin(m*pi*x) * sin(n*pi*y))  # Ih v (not normalized)
    return phi

# (1) Discrete eigenpairs on V (CG1)
problem = LinearEigenproblem(A, M, bcs=bcs, restrict=True)
solver_params = {"eps_smallest_magnitude": None, "eps_tol": 1e-12}
nev = 40  # a few extra to navigate degeneracies
eigs = LinearEigensolver(problem, n_evals=nev, solver_parameters=solver_params)
nconv = eigs.solve()

lam_h = []
vh_list = []
for i in range(min(nconv, nev)):
    lam_h.append(eigs.eigenvalue(i))
    vr, vi = eigs.eigenfunction(i)
    vh = Function(V, name=f"vh_{i}")
    vh.assign(vr)
    nrm = np.sqrt(assemble(inner(vh, vh) * dx))
    if nrm > 0:
        vh.assign(vh / nrm)  # L2-normalize vh
    vh_list.append(vh)

# (2) Match exact modes to discrete ones by max L2 correlation (handles multiplicity)
used = set()
def match_v_to_vh(v_exact):
    best_i, best_c = -1, -1.0
    for i, vh in enumerate(vh_list):
        if i in used:
            continue
        c = float(assemble(inner(v_exact, vh) * dx))
        if abs(c) > abs(best_c):
            best_i, best_c = i, c
    used.add(best_i)
    vh = vh_list[best_i].copy(deepcopy=True)
    if best_c < 0:
        vh.assign(-vh)  # align sign
    return best_i, lam_h[best_i], vh

# (3) Compute errors and print
ncheck = 20
rows = []
print("Using exact v:")
print("\n# k  (m,n)   error_exact           error_predicted        |diff|")
for k, (lam_exact, m, n) in enumerate(exact_list[:ncheck], start=1):
    v_exact = build_exact_v(m, n, Vd)  # exact in rich space, normalized
    i, lamh, vh = match_v_to_vh(v_exact)
    phi_h = build_phi_h(m, n)          # Ih v in V

    # sigma_h = 1/2 ||v - vh||^2   (v and vh are both L2-normalized)
    e = Function(Vd).interpolate(v_exact - vh)
    sigma_h = 0.5 * assemble(inner(e, e) * dx)

    # RHS = a(vh, v - phi_h) - lamh * (vh, v - phi_h)
    rhs = assemble(inner(grad(vh), grad(v_exact - phi_h)) * dx) \
          - lamh * assemble(inner(vh, v_exact - phi_h) * dx)

    error_exact = lam_exact - lamh
    denom = 1.0 - sigma_h
    error_pred = rhs / denom if abs(denom) > 1e-14 else float("nan")
    diff = abs(error_exact - error_pred)

    rows.append((k, m, n, error_exact, error_pred, diff))
    print(f"{k:2d}  ({m},{n})  {error_exact: .6e}   {error_pred: .6e}   {diff: .2e}")

# (optional) keep rows in a numpy array or write CSV
# np.savetxt("eigen_error_compare.csv", np.array(rows, dtype=object), fmt="%s", delimiter=",")

# Spaces and forms for degree p+1
Vp1  = FunctionSpace(mesh, "CG", dual_degree)
up1  = TrialFunction(Vp1); vp1 = TestFunction(Vp1)
Ap1  = inner(grad(up1), grad(vp1)) * dx
Mp1  = inner(up1, vp1) * dx
bcs_p1 = [DirichletBC(Vp1, 0.0, "on_boundary")]

# ---------- helpers ----------
def solve_hermitian_eigs(Aform, Mform, Vspace, bcs, nev=20):
    """Solve a(v,w)=lambda (v,w) for the smallest eigenpairs; return (lams, [vh_i])."""
    problem = LinearEigenproblem(Aform, Mform, bcs=bcs, restrict=True)
    eigs = LinearEigensolver(problem, n_evals=nev,
                             solver_parameters={"eps_gen_hermitian": None,
                                                "eps_smallest_magnitude": None,
                                                "eps_tol": 1e-12})
    nconv = eigs.solve()
    lams, vecs = [], []
    for i in range(min(nconv, nev)):
        lams.append(eigs.eigenvalue(i))
        vr, vi = eigs.eigenfunction(i)       # real/imag; imag ~ 0 here
        vh = Function(Vspace)
        vh.assign(vr)
        # L2-normalize
        nrm = assemble(inner(vh, vh) * dx)**0.5
        if nrm > 0:
            vh.assign(vh / nrm)
        vecs.append(vh)
    return lams, vecs

def best_match(target, pool, used):
    """Pick index in pool (not in used) with max |(target, pool_i)|; align sign."""
    best_i, best_c = -1, 0.0
    for i, v in enumerate(pool):
        if i in used:
            continue
        c = float(assemble(inner(target, v) * dx))
        if abs(c) > abs(best_c):
            best_i, best_c = i, c
    if best_i < 0:
        raise RuntimeError("No available vector to match.")
    vh = pool[best_i].copy(deepcopy=True)
    if best_c < 0:
        vh.assign(-vh)
    used.add(best_i)
    return best_i, vh, best_c

# ---------- solve eigenproblems on V (p) and V_{p+1} ----------
nev = 40  # a few extra to ride out degeneracies
lam_h,  Vp_vecs  = solve_hermitian_eigs(A,  M,  V,  bcs,    nev=nev)
lam_hp1, Vp1_vecs = solve_hermitian_eigs(Ap1, Mp1, Vp1, bcs_p1, nev=nev)

# ---------- build analytic modes once for robust indexing by (m,n) ----------
x, y = SpatialCoordinate(mesh)
def exact_expr(m, n):
    return sin(m*pi*x) * sin(n*pi*y)

# ---------- compute and print for first n=10 modes ----------
ncheck = 20
used_p1 = set()
used_p  = set()

print("Approximating v in CG(p+1):")
print("\n# k  (m,n)    error_exact     error_predicted      |diff|")
rows = []
for k, (lam_exact, m, n) in enumerate(exact_list[:ncheck], start=1):
    # 1) Build analytic v_{m,n} just for matching order (no longer used in estimator)
    v_ref = Function(Vd)
    v_ref.interpolate(exact_expr(m, n))
    # normalize v_ref to be comparable
    nrm_ref = assemble(inner(v_ref, v_ref) * dx)**0.5
    if nrm_ref > 0:
        v_ref.assign(v_ref / nrm_ref)

    # 2) Pick the V_{p+1} eigenvector that best matches v_ref
    idx_p1, v_hp1, _ = best_match(v_ref, Vp1_vecs, used_p1)

    # 3) With that v_hp1 fixed, pick the V (degree p) eigenvector that best matches v_hp1
    idx_p,  v_h,   _ = best_match(v_hp1, Vp_vecs, used_p)

    lamh   = lam_h[idx_p]         # degree-p eigenvalue used in "exact" error
    # lamhp1 = lam_hp1[idx_p1]     # (available if you also want hp1-vs-h comparisons)

    # 4) Build φ_h = I_h v^{p+1} in V (nodal interpolant)
    phi_h = interp(v_hp1, V)

    # 5) σ_h = 1/2 ||v^{p+1} - v_h||^2  (both normalized)
    e = Function(Vd).interpolate(v_hp1 - v_h)
    sigma_h = 0.5 * assemble(inner(e, e) * dx)

    # 6) RHS = a(v_h, v^{p+1} - φ_h) - λ_h (v_h, v^{p+1} - φ_h)
    rhs = assemble(inner(grad(v_h), grad(v_hp1 - phi_h)) * dx) \
        - lamh * assemble(inner(v_h, (v_hp1 - phi_h)) * dx)

    # 7) error_exact and error_predicted
    error_exact = lam_exact - lamh
    denom = 1.0 - sigma_h
    error_pred = rhs / denom if abs(denom) > 1e-14 else float("nan")
    diff = abs(error_exact - error_pred)

    rows.append((k, m, n, error_exact, error_pred, diff))
    print(f"{k:2d}  ({m:1d},{n:1d})   {error_exact: .6e}                 {error_pred: .6e}                 {diff: .2e}")


# ----------------------------
# (Optional) sanity check: with phi_h = v_h the identity collapses to normalization
# ----------------------------
# for k, (lam_exact, m, n) in enumerate(exact_list[:3], start=1):
#     v_exact = build_exact_v(m, n)
#     idx, lamh, vh, corr = match_v_to_discrete(v_exact)  # will skip since used; comment "used" logic if you want to test
#     phi_h = vh
#     e = Function(Vd).interpolate(v_exact - vh)
#     sigma_h = 0.5 * assemble(inner(e, e) * dx)
#     lhs = (lam_exact - lamh) * (1.0 - sigma_h)
#     rhs = assemble(inner(grad(vh), grad(v_exact - phi_h)) * dx) - lamh * assemble(inner(vh, v_exact - phi_h) * dx)
#     print("(phi_h=vh) residual:", float(lhs - rhs))
