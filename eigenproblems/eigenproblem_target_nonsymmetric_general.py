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
nx = 10
degree = 1
NEV_SOLVE = 15

# Pick a target by (m,n) OR set 'target' directly to a float
m_t, n_t = 3, 3
target = float(np.pi**2 * (m_t*m_t + n_t*n_t))  # override this with a number if you like
target = 39.804171910300276
N_SHOW = 1                                     # number of closest eigenpairs to list
print("Target eigenvaue: ", target)
# Extra compute slack (compute more than you show, to be safe around degeneracies)
NEV_SOLVE = 15

# Mesh and spaces
#mesh = Mesh(unit_square.GenerateMesh(maxh=1/nx))
mesh = PeriodicSquareMesh(nx,nx,1.0)
V  = FunctionSpace(mesh, "CG", degree)
u  = TrialFunction(V); v = TestFunction(V)
vel = as_vector([1, 0.3])
eps= Constant(0.02)
A  = inner(grad(u), grad(v)) * dx + v * dot(vel, grad(u))*dx
M  = inner(u, v) * dx
#bcs = [DirichletBC(V, 0.0, "on_boundary")]
bcs = []
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

(v_old,u_old) = A.arguments()
A_high = ufl.replace(A, {v_old: v_high, u_old: u_high})
M_high = ufl.replace(M, {v_old: v_high, u_old: u_high})
bcs_high = reconstruct_bcs(bcs, V_high)

(v_old,u_old) = A_adj.arguments()
A_adj_high  = ufl.replace(A_adj, {v_old: v_high, u_old: u_high})

# Eigensolve helpers
def l2_normalize(f):
    nrm = assemble(inner(f, f)*dx)**0.5
    if nrm > 0:
        f.assign(f/nrm)
    return f

def l2_normalize_complex(vr, vi):
    n2 = assemble(inner(vr, vr)*dx + inner(vi, vi)*dx)
    n  = n2**0.5
    if n > 0:
        vr.assign(vr/n)
        vi.assign(vi/n)
    return vr, vi

# --- complex-safe helpers built from your forms A, M ---

def complex_split(u):
    """Accept Function or (real, imag) tuple -> (ur, ui) Functions.
       If only real is given, ui is a zero Function in the same space."""
    if isinstance(u, tuple) or isinstance(u, list):
        ur, ui = u
    else:
        ur, ui = u, None
    if ui is None:
        ui = Function(ur.function_space())  # zero by default
    return ur, ui

def complex_inner(u, w, form):
    """⟨u, w⟩_form using the bilinear form 'form' (e.g. M or A). Returns complex."""
    ur, ui = complex_split(u)
    wr, wi = complex_split(w)
    re_form = action(action(form, ur), wr) + action(action(form, ui), wi)
    im_form = action(action(form, ur), wi) - action(action(form, ui), wr)
    re = assemble(re_form)
    im = assemble(im_form)
    return re + 1j*im

def complex_norm(u, Mform):
    """‖u‖ induced by Mform (L2 if M = inner(u,v)dx)."""
    ur, ui = complex_split(u)
    val = assemble(action(action(Mform, ur), ur) + action(action(Mform, ui), ui))
    return val**0.5

def complex_rhs(Aform, Mform, u_h, e, lam):
    """Complex residual: a(u_h,e) - λ (u_h,e)."""
    return complex_inner(u_h, e, Aform) - lam * complex_inner(u_h, e, Aform)

def solve_eigs(Aform, Mform, Vspace, bcs, nev, solver_parameters):
    prob = LinearEigenproblem(Aform, Mform, bcs=bcs, restrict=True)
    es = LinearEigensolver(prob, n_evals=nev, solver_parameters=solver_parameters)
    nconv = es.solve()
    lam, vecs = [], []
    for i in range(min(nconv, nev)):
        lam.append(es.eigenvalue(i))
        vr, vi = es.eigenfunction(i)
        vecs.append(l2_normalize_complex(vr,vi))
    return lam, vecs


def phase_align(eigfunc, target, Mform):
    """
    Rotate eigfunc by e^{-i arg(<target,eigfunc>_M)} so <target, aligned> is real >= 0.
    Returns (wr_aligned, wi_aligned, abs_inner).
    """
    tr, ti = complex_split(target)
    wr, wi = complex_split(eigfunc)
    c = complex_inner((tr, ti), (wr, wi), Mform)
    if c == 0:
        return wr, wi, 0.0
    phase = c.conjugate() / abs(c)   # = cosθ + i sinθ
    cs, sn = phase.real, phase.imag
    wr2 = wr.copy(deepcopy=True); wi2 = wi.copy(deepcopy=True)
    wr2.assign(cs*wr + sn*wi)
    wi2.assign(-sn*wr + cs*wi)
    return wr2, wi2, abs(c)

# --- corrected matcher (uses Mform; prints top-N; returns aligned original) ---
def match_best_complex(target, candidates_pairs, lambdas, Mform, print_top=5):
    """
    target: Function or (r,i); candidates_pairs: list of (wr,wi) (wi may be zero Function).
    Returns (lambda_or_index, (wr_aligned, wi_aligned)).
    """
    nt = complex_norm(target, Mform)
    scores = []
    for i, cand in enumerate(candidates_pairs):
        nw = complex_norm(cand, Mform)
        if nw == 0:
            continue
        c = complex_inner(target, cand, Mform)
        scores.append((abs(c)/(nt*nw), i, c))
    scores.sort(key=lambda t: t[0], reverse=True)
    if not scores:
        raise RuntimeError("No nonzero candidate matched.")

    # print top correlations
    for r, (corr, i, c) in enumerate(scores[:print_top], start=1):
        lam_i = lambdas[i] if lambdas is not None else None
        print(f"{r:2d}) idx={i:3d}, corr={corr:.16f}, |<t,w>|={abs(c):.3e}, λ={lam_i}")

    # best, aligned in ORIGINAL space
    _, best_i, best_c = scores[0]
    wr_al, wi_al, _ = phase_align(candidates_pairs[best_i], target, Mform)
    eig = (lambdas[best_i] if lambdas is not None else best_i)
    return eig, (wr_al, wi_al)


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
    lam, _ = solve_eigs(A, M, V, bcs, nev=2, solver_parameters=params)
    return lam

print("λ_max(discrete) ≈", lambda_max_discrete()) # Doesn't work yet.

solver_parameters_target = {
    #"eps_gen_hermitian": None,
    "eps_target": target
}
lamh_prim_vec,  eigfunc_prim_vec   = solve_eigs(A,  M,  V,  bcs,    nev=NEV_SOLVE, solver_parameters=solver_parameters_target)
norm = complex_norm(eigfunc_prim_vec[0],M)
print(norm)
norm = assemble(inner(eigfunc_prim_vec[0][0],eigfunc_prim_vec[0][0])*dx)**0.5
print(norm)

lamh_prim_high_vec, eigfunc_prim_high_vec = solve_eigs(A_high, M_high, V_high, bcs_high, nev=NEV_SOLVE, solver_parameters=solver_parameters_target)
lamh_adj_vec, eigfunc_adj_vec = solve_eigs(A_adj, M, V_high, bcs_high, nev=NEV_SOLVE, solver_parameters=solver_parameters_target)
lamh_adj_high_vec, eigfunc_adj_high_vec = solve_eigs(A_adj_high, M_high, V_high, bcs_high, nev=NEV_SOLVE, solver_parameters=solver_parameters_target)

eigfunc_prim = eigfunc_prim_vec[0]
lamh_prim = lamh_prim_vec[0]
print("Computed eigenvalue: ",lamh_prim)
print("Matching eigenfunctions [primal in V_high]...")
lamh_prim_high, eigfunc_prim_high = match_best_complex(eigfunc_prim, eigfunc_prim_high_vec , lamh_prim_high_vec, M)
print("Matching eigenfunctions [dual in V]...")
lamh_adj, eigfunc_adj = match_best_complex(eigfunc_prim,eigfunc_adj_vec,lamh_adj_vec, M)
print("Matching eigenfunctions [dual in V_high]...")
lamh_adj_high, eigfunc_adj_high = match_best_complex(eigfunc_prim,eigfunc_adj_high_vec,lamh_adj_high_vec, M)

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

def automatic_error_indicators_complex(weight, F_re, F_im):
    """
    weight: Function or (real, imag) tuple for the DWR weight (e.g. u_{p+1}-I_V u_{p+1})
    F_re, F_im: real-valued residual forms from build_*_residual_forms(...)
    Returns a DG0 Function with RMS-combined per-cell indicators.
    """
    wr, wi = complex_split(weight)
    eta_re = automatic_error_indicators(wr, F_re)  # your existing real routine
    eta_im = automatic_error_indicators(wi, F_im)

    DG0 = eta_re.function_space()
    eta = Function(DG0, name="eta_complex")
    with eta_re.dat.vec_ro as vr, eta_im.dat.vec_ro as vi, eta.dat.vec as vo:
        ar = vr.getArray().copy(); ai = vi.getArray().copy()
        vo.setArray((ar*ar + ai*ai)**0.5)
    return eta


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

def manual_error_indicators_primal_complex(u_h, e, lamh):
    """
    Per-cell indicators for the primal residual weighted by e.
    u_h, e may be Function or (real, imag) tuple. lamh may be complex.
    """
    ur, ui = complex_split(u_h)
    er, ei = complex_split(e)
    lamr, lami = lamh.real, lamh.imag

    n_f  = FacetNormal(mesh)
    DG0  = FunctionSpace(mesh, "DG", degree=0)
    test = TestFunction(DG0)

    # real component
    eta_re = assemble(
          inner(div(grad(ur)), er*test)*dx
        - lamr*(ur*er)*test*dx + lami*(ui*er)*test*dx
        + (jump(grad(ur), n_f)*er)*0.5*both(test)*dS
        + inner(dot(-grad(ur), n_f), er*test)*ds
    )
    # imag component
    eta_im = assemble(
          inner(div(grad(ui)), ei*test)*dx
        - lamr*(ui*ei)*test*dx - lami*(ur*ei)*test*dx
        + (jump(grad(ui), n_f)*ei)*0.5*both(test)*dS
        + inner(dot(-grad(ui), n_f), ei*test)*ds
    )

    # RMS combine
    eta = Function(DG0, name="eta_primal")
    with eta_re.dat.vec_ro as vr, eta_im.dat.vec_ro as vi, eta.dat.vec as vo:
        ar = vr.getArray().copy(); ai = vi.getArray().copy()
        vo.setArray((ar*ar + ai*ai)**0.5)
    return eta

def manual_error_indicators_adjoint_complex(z_h, w, lamh):
    """
    Per-cell indicators for the adjoint residual weighted by w.
    L*(v) = a(v,z_h) - conj(λ_h)(v,z_h).
    """
    zr, zi = complex_split(z_h)
    wr, wi = complex_split(w)
    lamr, lami = lamh.real, lamh.imag

    n_f  = FacetNormal(mesh)
    DG0  = FunctionSpace(mesh, "DG", degree=0)
    test = TestFunction(DG0)

    # real component
    eta_re = assemble(
          inner(div(grad(zr)), wr*test)*dx
        - lamr*(zr*wr)*test*dx - lami*(zi*wr)*test*dx
        + (jump(grad(zr), n_f)*wr)*0.5*both(test)*dS
        + inner(dot(-grad(zr), n_f), wr*test)*ds
    )
    # imag component
    eta_im = assemble(
          inner(div(grad(zi)), wi*test)*dx
        - lamr*(zi*wi)*test*dx + lami*(zr*wi)*test*dx
        + (jump(grad(zi), n_f)*wi)*0.5*both(test)*dS
        + inner(dot(-grad(zi), n_f), wi*test)*ds
    )

    eta = Function(DG0, name="eta_adjoint")
    with eta_re.dat.vec_ro as vr, eta_im.dat.vec_ro as vi, eta.dat.vec as vo:
        ar = vr.getArray().copy(); ai = vi.getArray().copy()
        vo.setArray((ar*ar + ai*ai)**0.5)
    return eta

def build_adjoint_residual_forms_from_forms(A, M, z_h, lamh):
    """
    Build L*(v) = a(v, z_h) - conj(lamh) (v, z_h) as two real linear forms:
        F_re + i F_im
    using only the bilinear forms A and M.
    Returns: (F_re, F_im), both linear forms in the test function space of A/M.
    """
    zr, zi = complex_split(z_h)          # z_h = zr + i zi (Functions)
    lamr, lami = lamh.real, lamh.imag    # conj(lamh) = lamr - i lami

    # a(v, z) = a(v, zr) + i a(v, zi) ;   (v, z) = (v, zr) + i (v, zi)
    # => Re: a(v,zr) - lamr (v,zr) - lami (v,zi)
    #    Im: a(v,zi) - lamr (v,zi) + lami (v,zr)

    F_re = action(A, zr) - lamr*action(M, zr) - lami*action(M, zi)
    F_im = action(A, zi) - lamr*action(M, zi) + lami*action(M, zr)
    return F_re, F_im



def run_targeted():
    print("   λ_h                 sigma_h     error_exact     error_predicted   effectivity1   sum(local errors)   effectivity2")

    # unpack the matched eigenpairs you already computed
    u_h   = eigfunc_prim            # (ur, ui)
    u_p   = eigfunc_prim_high       # (upr, upi)
    z_h   = eigfunc_adj             # (zr, zi)
    z_p   = eigfunc_adj_high        # (zpr, zpi)
    lamh  = lamh_prim               # complex scalar ok
    lamh_adj_loc = lamh_adj         # if you solved adjoint; else use lamh.conjugate()

    # build I_V(u_p) componentwise, then e = u_p - I_V u_p  (dual weight)
    upr, upi = complex_split(u_p)
    phi_hr = Function(V); phi_hr.interpolate(upr)
    phi_hi = Function(V); phi_hi.interpolate(upi)
    e = (upr - phi_hr, upi - phi_hi)

    # errors between p+1 and p (primal and adjoint)
    uhr, uhi = complex_split(u_h)
    zpr, zpi = complex_split(z_p)
    zhr, zhi = complex_split(z_h)

    e_sigma    = (upr - uhr,  upi - uhi)     # primal error
    e_sigma_ad = (zpr - zhr,  zpi - zhi)     # adjoint error

    # ----- sigma_h and rhs (complex-safe) -----
    # Self-adjoint: σ_h = 1/2 ||u_p - u_h||_M^2, rhs = a(u_h,e) - λ_h (u_h,e)
    # Non-Hermitian: symmetric DWR: average primal- and adjoint-residual contributions
    self_adjoint = False

    if self_adjoint:
        sigma_h = 0.5 * complex_inner(M, e_sigma, e_sigma).real
        rhs_c   = complex_inner(A, u_h, e) - lamh * complex_inner(M, u_h, e)
    else:
        # primal residual weighted by adjoint error
        rhs_pr  = complex_inner(A, u_h, e_sigma_ad) - lamh * complex_inner(M, u_h, e_sigma_ad)
        # adjoint residual weighted by primal weight  (use adjoint eigenvalue if available)
        lam_adj_use = lamh_adj_loc if lamh_adj_loc is not None else lamh.conjugate()
        rhs_ad  = complex_inner(A_adj, z_h, e) - lam_adj_use * complex_inner(M, z_h, e)
        rhs_c   = 0.5 * (rhs_pr + rhs_ad)
        # mixed σ_h (real part)
        sigma_h = 0.5 * complex_inner(M, e_sigma, e_sigma_ad).real

    denom = 1.0 - sigma_h
    error_pred = abs(rhs_c) / denom if abs(denom) > 1e-14 else float("nan")

    # exact error against your "target" if you have one
    lam_exact   = target
    error_exact = abs(lam_exact - lamh)
    effectivity = (error_pred / error_exact) if error_exact > 0 else float("nan")

    # ----- Local indicators -----
    # Only use the existing real-valued routines if the mode is numerically real.
    is_real_mode = (assemble(inner(uhi, uhi)*dx) < 1e-14) and (abs(lamh.imag) < 1e-14)

    if is_real_mode:
        # manual indicator with real parts
        e_real = e[0]
        eta_T = manual_error_indicators(uhr, e_real, lamh.real)
        with eta_T.dat.vec_ro as vvec:
            eta_array = vvec.getArray().copy()
        total = np.abs(eta_array).sum()
        total_normalised = total/denom if abs(denom) > 1e-14 else float("nan")
        eff2 = (total_normalised / error_exact) if error_exact > 0 else float("nan")

        # automatic indicator with real parts
        vT = TestFunction(V)
        form_re = inner(grad(uhr), grad(vT))*dx - lamh.real*inner(uhr, vT)*dx
        eta_T_auto = automatic_error_indicators(e_real, form_re)
        with eta_T_auto.dat.vec_ro as vvec:
            eta_array_auto = vvec.getArray().copy()
        total_auto = np.abs(eta_array_auto).sum()
        total_normalised_auto = total_auto/denom if abs(denom) > 1e-14 else float("nan")
        eff2_auto = (total_normalised_auto / error_exact) if error_exact > 0 else float("nan")
    else:
        # skip local indicators for complex modes (needs Re/Im split + RMS combine)
        total_normalised = float("nan"); eff2 = float("nan")
        total_normalised_auto = float("nan"); eff2_auto = float("nan")

    # ----- print -----
    lam_str = f"{lamh.real:.6f}+{lamh.imag:.6f}j" if abs(lamh.imag) > 0 else f"{lamh.real:.6f}"
    print(f"{lam_str:>18}   {sigma_h: .6f}   {error_exact: .6e}     {error_pred: .6e}    {effectivity: .3f}     {total_normalised: .6e}      {eff2: .3f} ")
    print(f"Automatic:                                                                         {total_normalised_auto: .6e}      {eff2_auto: .3f} ")

    return {
        "lamh": lamh, "sigma_h": sigma_h, "error_exact": error_exact,
        "error_pred": error_pred, "effectivity": effectivity
    }

sys.exit()
# Targeted run: list nearest computed eigenpairs (with effectivities)
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
        rhs = assemble(residual(action(A,u_h),e)) - lamh * assemble(residual(action(M,u_h),e))
    else:
        e_sigma_adj = z_p - z_h
        sigma_h = 0.5 * assemble(inner(e_sigma, e_sigma_adj) * dx)
        e_adj = z_p - z_h
        rhs = 0.5*( assemble(residual(action(A,u_h),e_adj)) - lamh * assemble(residual(action(M,u_h),e_adj))
                   + assemble(residual(action(A_adj,z_h),e)) - lamh * assemble(residual(action(M,z_h),e))
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
    vT = TestFunction(V)
    form = inner(grad(u_h), grad(vT)) * dx - lamh * inner(u_h, vT) * dx
    form = action(A,u_h)
    eta_T_auto = automatic_error_indicators(e, form)
    with eta_T_auto.dat.vec_ro as vvec:
        eta_array_auto = vvec.getArray().copy()
    total_auto = np.abs(eta_array_auto).sum()
    total_normalised_auto = total_auto/denom if abs(denom) > 1e-14 else float("nan")
    eff2_auto = (total_normalised_auto / error_exact) if error_exact > 0 else float("nan")

    # print in your original format
    print(f"{lamh: .6f}   {sigma_h: .6f}   {error_exact: .6e}     {error_pred: .6e}    {effectivity: .3f}     {total_normalised: .6e}      {eff2: .3f} ")
    print(f"Automatic:                                                                         {total_normalised_auto: .6e}      {eff2_auto: .3f} ")

rows_target = run_targeted()

sys.exit()

# Identify the analytic (m,n) matching a function
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