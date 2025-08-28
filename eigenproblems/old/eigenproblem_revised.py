from firedrake import *
from firedrake.eigensolver import LinearEigenproblem, LinearEigensolver
from netgen.occ import *
import numpy as np

# ----------------------------
# Mesh and spaces (yours)
# ----------------------------
nx = 10
mesh = Mesh(unit_square.GenerateMesh(maxh=1/nx))
# mesh = UnitSquareMesh(nx, nx)

degree = 1
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

def build_exact_v(m, n):
    v_exact = Function(Vd, name=f"v_{m}_{n}")
    v_exact.interpolate(sin(m*pi*x) * sin(n*pi*y))
    # L2-normalize v
    nm = np.sqrt(assemble(inner(v_exact, v_exact) * dx))
    if nm > 0:
        v_exact.assign(v_exact / nm)
    return v_exact

def build_phi_h(m, n):
    phi_h = Function(V, name=f"phi_{m}_{n}")
    phi_h.interpolate(sin(m*pi*x) * sin(n*pi*y))  # nodal interpolant Ih v
    return phi_h

# ----------------------------
# Discrete eigenpairs on V (CG1) using SLEPc via Firedrake
# ----------------------------
# Build generalized Hermitian eigenproblem: find (lambda_h, v_h) with a(v_h, w)=lambda_h (v_h, w)
problem = LinearEigenproblem(A, M, bcs=bcs, restrict=True)
solver_params = {
    "eps_gen_hermitian": None,      # symmetric generalized EVP
    "eps_smallest_magnitude": None, # get the smallest positive eigenvalues
    "eps_tol": 1e-12
}
nev = 20  # compute a few extra to handle degeneracies cleanly
eigs = LinearEigensolver(problem, n_evals=nev, solver_parameters=solver_params)
nconv = eigs.solve()

# Collect eigenvalues/eigenfunctions and L2-normalize v_h
lam_h = []
vh_list = []
for i in range(min(nconv, nev)):
    lam_h.append(eigs.eigenvalue(i))
    vr, vi = eigs.eigenfunction(i)  # real/imag parts (imag should be ~0 here)
    vh = Function(V, name=f"vh_{i}")
    vh.assign(vr)
    nm = np.sqrt(assemble(inner(vh, vh) * dx))
    if nm > 0:
        vh.assign(vh / nm)
    vh_list.append(vh)

# ----------------------------
# Match each exact mode to the best discrete vector (handles degeneracy)
# ----------------------------
used = set()

def match_v_to_discrete(v_exact):
    best_i, best_c, best_abs = -1, 0.0, -1.0
    for i, vh in enumerate(vh_list):
        if i in used:
            continue
        c = assemble(inner(v_exact, vh) * dx)  # L2 correlation
        if abs(c) > best_abs:
            best_abs, best_c, best_i = abs(c), c, i
    if best_i == -1:
        raise RuntimeError("No available discrete eigenvector left to match.")
    used.add(best_i)
    vh = vh_list[best_i].copy(deepcopy=True)
    # Align sign to maximize correlation
    if best_c < 0:
        vh.assign(-vh)
    return best_i, lam_h[best_i], vh, float(best_abs)

# ----------------------------
# Evaluate the identity for the first n=10 exact pairs
# ----------------------------
ncheck = 10
results = []  # store rows for later inspection
print("\n--- Error identity (phi_h = Ih v) for first n=10 modes ---")
for k, (lam_exact, m, n) in enumerate(exact_list[:ncheck], start=1):
    v_exact = build_exact_v(m, n)   # exact, L2-normalized
    idx, lamh, vh, corr = match_v_to_discrete(v_exact)
    phi_h = build_phi_h(m, n)       # Ih v in V (not normalized)

    # sigma_h = 1/2 ||v - v_h||_0^2  (both v, v_h are L2-normalized)
    e = Function(Vd, name=f"e_{m}_{n}")
    e.interpolate(v_exact - vh)
    sigma_h = 0.5 * assemble(inner(e, e) * dx)

    # LHS and RHS of (45)
    lhs = (lam_exact - lamh) * (1.0 - sigma_h)
    rhs = assemble(inner(grad(vh), grad(v_exact - phi_h)) * dx) \
          - lamh * assemble(inner(vh, (v_exact - phi_h)) * dx)

    res = lhs - rhs   # should be ~ 0 (discretization/quad errors)
    results.append((m, n, lam_exact, lamh, lam_exact - lamh, sigma_h, float(lhs), float(rhs), float(res), corr))
    print(f"[{k:2d}] (m,n)=({m},{n}) "
          f"λ={lam_exact:.8f}  λ_h={lamh:.8f}  |Δ|={lam_exact-lamh:.3e}  "
          f"σ_h={sigma_h:.3e}  LHS-RHS={res:.3e}  corr={corr:.3f}")

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
