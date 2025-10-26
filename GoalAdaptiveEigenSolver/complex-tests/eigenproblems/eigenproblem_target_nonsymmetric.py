# --- put this at the very top (before importing firedrake) to kill the OMP warning ---
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

from firedrake import *
from firedrake.eigensolver import LinearEigenproblem, LinearEigensolver
from netgen.occ import *
import numpy as np
from firedrake.__future__ import interpolate as interp  # symbolic interpolate
import sys

# User knobs
nx = 20
degree = 1
NEV_SOLVE = 15

# Targeted run: list nearest computed eigenpairs (with effectivities)
def replace_trial(bilinear_form, coeff):
    test, trial = bilinear_form.arguments()
    linear_form = replace(bilinear_form, {trial: coeff})
    return linear_form

def replace_test(bilinear_form, coeff):
    test, trial = bilinear_form.arguments()
    linear_form = replace(bilinear_form, {test: coeff})
    return linear_form

def replace_both(bilinear_form, trial_coeff, test_coeff):
    test, trial = bilinear_form.arguments()
    return replace(bilinear_form, {test: test_coeff, trial: trial_coeff})

# Pick a target by (m,n) OR set 'target' directly to a float
m_t, n_t = 3, 3
target = float(np.pi**2 * (m_t*m_t + n_t*n_t))  # override this with a number if you like
N_SHOW = 1                                     # number of closest eigenpairs to list
print("Target eigenvaue: ", target)
# Extra compute slack (compute more than you show, to be safe around degeneracies)
NEV_SOLVE = 15

# Mesh and spaces
mesh = Mesh(unit_square.GenerateMesh(maxh=1/nx))

V  = FunctionSpace(mesh, "CG", degree)
u  = TrialFunction(V); v = TestFunction(V)
A  = inner(grad(u), grad(v)) * dx
M  = inner(u,v) * dx
bcs = [DirichletBC(V, 0.0, "on_boundary")]
A_adj = adjoint(A)

def reconstruct_bc_value(bc, V):
    if not isinstance(bc._original_arg, firedrake.Function):
        return bc._original_arg
    return Function(V).interpolate(bc._original_arg)

def reconstruct_bcs(bcs, V):
    """Reconstruct a list of bcs"""
    new_bcs = []
    for bc in bcs:
        V_ = V
        for index in bc._indices:
            V_ = V_.sub(index)
        g = reconstruct_bc_value(bc, V_)
        new_bcs.append(bc.reconstruct(V=V_, g=g))
    return new_bcs

dual_extra_degree = 1
element = V.ufl_element()
high_degree = degree + dual_extra_degree # By default use dual degree
high_element = PMGPC.reconstruct_degree(element, high_degree)
V_high = FunctionSpace(mesh, high_element)
u_high = TrialFunction(V_high)
v_high = TestFunction(V_high)

A_high = replace_both(A, v_high, u_high)
M_high = replace_both(M, v_high, u_high)
bcs_high = reconstruct_bcs(bcs, V_high)
A_adj_high  = replace_both(A_adj, v_high, u_high)

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

# Eigensolve helpers
def solve_eigs(Aform, Mform, bcs, nev, solver_parameters):
    prob = LinearEigenproblem(Aform, Mform, bcs=bcs, restrict=True)
    es = LinearEigensolver(prob, n_evals=nev, solver_parameters=solver_parameters)
    nconv = es.solve()
    lam, vecs = [], []
    for i in range(min(nconv, nev)):
        lam.append(es.eigenvalue(i))
        vr, vi = es.eigenfunction(i)
        vecs.append(l2_normalize(vr))
    return lam, vecs

def match_best(target, candidates, lambdas=None):
    # target: Function; candidates: list[Function]; lambdas: list[...] or None
    N = 5  # how many to print after the best
    nt = float(assemble(inner(target, target) * dx))**0.5
    scores = []
    for i, w in enumerate(candidates):
        nw = float(assemble(inner(w, w) * dx))**0.5
        if nw == 0.0:
            continue
        c = complex(assemble(inner(target, w) * dx))
        corr = abs(c) / (nt * nw)
        scores.append((corr, i, c))

    scores.sort(key=lambda t: t[0], reverse=True)

    if not scores:
        raise RuntimeError("No nonzero candidate matched.")

    # best
    best_corr, best_i, best_c = scores[0]
    lam = lambdas[best_i] if lambdas is not None else None
    print(f" 1) idx={best_i:3d}, corr={best_corr:.6f}, <t,w>={best_c}, λ={lam}")

    # next N
    for r, (corr, i, c) in enumerate(scores[1:N+1], start=2):
        lam_i = lambdas[i] if lambdas is not None else None
        print(f"{r:2d}) idx={i:3d}, corr={corr:.6f}, <t,w>={c}, λ={lam_i}")

    # return phase-/sign-aligned copy in ORIGINAL space
    aligned = candidates[best_i].copy(deepcopy=True)
    if best_c != 0:
        phase = best_c.conjugate() / abs(best_c)
        aligned.assign(phase * aligned)

    return (lam if lambdas is not None else best_i), aligned

    
def lambda_max_discrete():
    params = {
        #"eps_gen_hermitian": None,
    }
    lam, _ = solve_eigs(A, M, bcs, nev=2, solver_parameters=params)
    return lam

print("λ_max(discrete) ≈", lambda_max_discrete()) # Doesn't work yet.

solver_parameters_target = {
    "eps_gen_hermitian": None,
    "eps_target": target
}

lamh_prim_vec,  eigfunc_prim_vec   = solve_eigs(A,  M, bcs,    nev=NEV_SOLVE, solver_parameters=solver_parameters_target)
lamh_prim_high_vec, eigfunc_prim_high_vec = solve_eigs(A_high, M_high, bcs_high, nev=NEV_SOLVE, solver_parameters=solver_parameters_target)
lamh_adj_vec, eigfunc_adj_vec = solve_eigs(A_adj, M, bcs_high, nev=NEV_SOLVE, solver_parameters=solver_parameters_target)
lamh_adj_high_vec, eigfunc_adj_high_vec = solve_eigs(A_adj_high, M_high, bcs_high, nev=NEV_SOLVE, solver_parameters=solver_parameters_target)

eigfunc_prim = eigfunc_prim_vec[0]
lamh_prim = lamh_prim_vec[0]
print("Computed eigenvalue: ",lamh_prim)
print("Matching eigenfunctions [primal in V_high]...")
lamh_prim_high, eigfunc_prim_high = match_best(eigfunc_prim, eigfunc_prim_high_vec , lamh_prim_high_vec)
print("Matching eigenfunctions [dual in V]...")
lamh_adj, eigfunc_adj = match_best(eigfunc_prim,eigfunc_adj_vec,lamh_adj_vec)
print("Matching eigenfunctions [dual in V_high]...")
lamh_adj_high, eigfunc_adj_high = match_best(eigfunc_prim,eigfunc_adj_high_vec,lamh_adj_high_vec)

# Residual-based indicators
def both(u):
    return u("+") + u("-")

def residual(form, test):  # Residual helper
    test_old = form.arguments()[0]
    return replace(form, {test_old: test})

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

def manual_error_indicators(v_h, e, lamh):
    n_f  = FacetNormal(mesh)
    DG0  = FunctionSpace(mesh, "DG", degree=0)
    test0 = TestFunction(DG0)
    eta_T = assemble(
            inner(div(grad(v_h)), e * test0) * dx
        + (-lamh * v_h * e) * test0 * dx
        + (jump(grad(v_h), n_f) * e) * 0.5*both(test0) * dS
        + inner(dot(-grad(v_h), n_f), e * test0) * ds
    )
    return eta_T

def run_targeted():
    print("   λ_h         sigma_h     error_exact     error_predicted   effectivity1   sum(local errors)   effectivity2")
    u_p = eigfunc_prim_high
    u_h = eigfunc_prim
    lamh = lamh_prim
    lam_exact = target

    z_p = eigfunc_adj_high
    z_h = eigfunc_adj
    lamh = lamh_prim

    phi_h = Function(V)
    phi_h.interpolate(eigfunc_prim_high)
    e = u_p - phi_h
    e_sigma = u_p - u_h

    self_adjoint = False
    # sigma_h = 1/2 ||u - u_h||^2
    if self_adjoint == True:
        sigma_h = 0.5 * assemble(inner(e_sigma, e_sigma) * dx)
        rhs = assemble( action(replace_trial(A,u_h),e) - lamh * replace_test(replace_trial(M,u_h),e)
                        )
    else:
        e_sigma_adj = z_p - z_h
        sigma_h = 0.5 * assemble(inner(e_sigma, e_sigma_adj) * dx)
        e_adj = z_p - z_h
        rhs = 0.5* assemble( replace_both(A,u_h,e_adj) - lamh * replace_both(M,u_h,e_adj)
                   + replace_both(A_adj,z_h,e) - lamh * replace_both(M,z_h,e)
                   )
    
    denom = 1.0 - sigma_h
    error_pred = abs(rhs / denom) if abs(denom) > 1e-14 else float("nan")

    # Exact error if possible
    error_exact = abs(lam_exact - lamh)
    effectivity = (error_pred / error_exact) if error_exact > 0 else float("nan")

    # ------- Local indicators (manual) -------
    eta_T = manual_error_indicators(u_h, e, lamh)
    with eta_T.dat.vec_ro as vvec:
        eta_array = vvec.getArray().copy()
    total = np.abs(eta_array).sum()
    total_normalised = total/denom if abs(denom) > 1e-14 else float("nan")
    eff2 = (total_normalised / error_exact) if error_exact > 0 else float("nan")

    # ------- Local indicators (automatic) -------
    form = replace_trial(A,u_h) - lamh * replace_trial(M, u_h)
    eta_T_auto = automatic_error_indicators(e, form)
    with eta_T_auto.dat.vec_ro as vvec:
        eta_array_auto = vvec.getArray().copy()
    total_auto = np.abs(eta_array_auto).sum()
    total_normalised_auto = total_auto/denom if abs(denom) > 1e-14 else float("nan")
    eff2_auto = (total_normalised_auto / error_exact) if error_exact > 0 else float("nan")

    # print in your original format
    print(f"{lamh: .6f}   {sigma_h: .6f}   {error_exact: .6e}     {error_pred: .6e}    {effectivity: .3f}     {total_normalised: .6e}      {eff2: .3f} ")
    print(f"Automatic:                                                                         {total_normalised_auto: .6e}      {eff2_auto: .3f} ")


# Run
rows_target = run_targeted()

sys.exit()


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
import matplotlib.pyplot as plt

# === utilities: put these once ===
def l2_norm(f):     return assemble(inner(f, f)*dx)**0.5
def h1s_norm(f):    return assemble(inner(grad(f), grad(f))*dx)**0.5
def l2_inner(f,g):  return assemble(inner(f, g)*dx)

def normalize(f):
    g = f.copy(deepcopy=True)
    n = l2_norm(g)
    if n > 0: g.assign(g/n)
    return g

def sign_align(f, ref):
    c = float(l2_inner(f, ref))
    if c < 0:
        g = f.copy(deepcopy=True); g.assign(-g); return g, -c
    return f, c

def to_space(f, Vtgt):
    # interpolate f into Vtgt (does nothing if already there)
    if isinstance(f, Function) and f.function_space() is Vtgt:
        return f
    g = Function(Vtgt); g.interpolate(f); return g

def compare_and_plot(u, u_hi, z, z_hi, *, Vplot=None, title=""):
    """
    u   : primal (in V)
    u_hi: primal (in V_high)
    z   : adjoint (in V or V_high)
    z_hi: adjoint (in V_high)
    Vplot: where to compare/plot (default: highest-order space among inputs)
    """
    # choose a plotting/compare space
    spaces = [f.function_space() for f in (u, u_hi, z, z_hi)]
    if Vplot is None:
        # pick the 'richest' one: assume higher degree has larger dim
        Vplot = max(spaces, key=lambda Vs: Vs.dim())

    # move everything to Vplot
    U    = to_space(u,    Vplot)
    Uhi  = to_space(u_hi, Vplot)
    Z    = to_space(z,    Vplot)
    Zhi  = to_space(z_hi, Vplot)

    # normalize all (L2), sign-align to primal U
    U    = normalize(U)
    Uhi  = normalize(Uhi)
    Z    = normalize(Z)
    Zhi  = normalize(Zhi)

    Uhi,  cUhi  = sign_align(Uhi, U)
    Z,    cZ    = sign_align(Z,   U)    # bi-orthogonal sign alignment (heuristic)
    Zhi,  cZhi  = sign_align(Zhi, U)

    # metrics
    def l2d(a,b):  return l2_norm(a-b)
    def h1d(a,b):  return h1s_norm(a-b)
    def corr(a,b): # L2-correlation in [-1,1]
        na, nb = l2_norm(a), l2_norm(b)
        return float(l2_inner(a,b)/(na*nb)) if na>0 and nb>0 else float("nan")

    print(f"== {title or 'Eigenfunction comparison'} (all in Vplot, L2-normalized) ==")
    print(f"corr(U, Uhi)  = {corr(U, Uhi): .6f}   L2|U-Uhi|={l2d(U,Uhi): .3e}   H1|U-Uhi|={h1d(U,Uhi): .3e}")
    print(f"corr(U, Z)    = {corr(U, Z):.6f}   L2|U-Z|  ={l2d(U,Z): .3e}   H1|U-Z|  ={h1d(U,Z): .3e}")
    print(f"corr(U, Zhi)  = {corr(U, Zhi): .6f}   L2|U-Zhi|={l2d(U,Zhi): .3e}   H1|U-Zhi|={h1d(U,Zhi): .3e}")

    # if you want an M-biorthogonality check (for generalized problems), after scaling Z so <Z, U>_M = 1:
    # s = l2_inner(Z, U);  if abs(s)>0: Z.assign(Z/s)  # now <Z, U>=1

    # common color scale
    vmin = min(float(U.dat.data_ro.min()),
            float(Uhi.dat.data_ro.min()),
            float(Z.dat.data_ro.min()),
            float(Zhi.dat.data_ro.min()))
    vmax = max(float(U.dat.data_ro.max()),
            float(Uhi.dat.data_ro.max()),
            float(Z.dat.data_ro.max()),
            float(Zhi.dat.data_ro.max()))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    titles = ["primal (V)", "primal (V+1)", "adjoint (V)", "adjoint (V+1)"]
    fields = [U, Uhi, Z, Zhi]

    for ax, f, t in zip(axes.flat, fields, titles):
        c = tripcolor(f, axes=ax, vmin=vmin, vmax=vmax)  # <- that's it
        ax.set_title(t)
        fig.colorbar(c, ax=ax)

    if title:
        fig.suptitle(title)
    plt.show()

#compare_and_plot(eigfunc_prim[0], eigfunc_prim_high, eigfunc_adj, eigfunc_adj_high, title=f"λ")