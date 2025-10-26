import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
from firedrake import *
from firedrake.eigensolver import LinearEigenproblem, LinearEigensolver
from netgen.occ import *
import numpy as np
from firedrake.__future__ import interpolate as interp  # symbolic interpolate

# Mesh and spaces
nx = 10
mesh = Mesh(unit_square.GenerateMesh(maxh=1/nx))

degree = 1
V  = FunctionSpace(mesh, "CG", degree)
u  = TrialFunction(V); v  = TestFunction(V)
A  = inner(grad(u), grad(v)) * dx
M  = inner(u, v) * dx
bcs = [DirichletBC(V, 0.0, "on_boundary")]

dual_degree = degree + 1
Vp1 = FunctionSpace(mesh, "CG", dual_degree)
up1 = TrialFunction(Vp1); vp1 = TestFunction(Vp1)
Ap1 = inner(grad(up1), grad(vp1))*dx
Mp1 = inner(up1, vp1)*dx
bcs_p1 = [DirichletBC(Vp1, 0.0, "on_boundary")]
nev = 200
ncheck = 80

# Automatic settings
residual_degree = 1

# Specific eigenvalue target
m_t, n_t = 3, 4
target = float(np.pi**2 * (m_t*m_t + n_t*n_t))

# Exact eigs
V_exact = FunctionSpace(mesh, "CG", degree + 8)  # richer space for exact modes & errors
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

def solve_eigs(Aform, Mform, Vspace, bcs, nev, solver_parameters):
    prob = LinearEigenproblem(Aform, Mform, bcs=bcs, restrict=True)
    es = LinearEigensolver(prob, n_evals=nev,
                           solver_parameters=solver_parameters)
    nconv = es.solve()
    lam, vecs = [], []
    for i in range(min(nconv, nev)):
        lam.append(es.eigenvalue(i))
        vr, vi = es.eigenfunction(i)
        #vh = Function(Vspace); vh.assign(vr)
        vecs.append(l2_normalize(vr))
    return lam, vecs

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

solver_parameters1 = {  "eps_gen_hermitian": None,
                        "eps_smallest_magnitude": None,
                        "eps_tol": 1e-12
                        }

solver_parameters_target = {"eps_gen_hermitian": None,
                            "eps_which": "target_magnitude",
                            "eps_target": target,
                            "st_type": "sinvert",
                            "st_ksp_type": "preonly",
                            "st_pc_type": "lu",                          # works out of the box; switch to hypre/gamg if you prefer
                            "st_pc_factor_mat_solver_type": "mumps",     # optional but fast/robust if available
                            "eps_tol": 1e-12,
                          }

# --- solve once per space ---
lam_h,  Vp_vecs  = solve_eigs(A, M, V, bcs, nev=nev, solver_parameters=solver_parameters1)
lam_hp1, Vp1_vecs = solve_eigs(Ap1, Mp1, Vp1, bcs_p1, nev=nev, solver_parameters=solver_parameters1)

def both(u):
    return u("+") + u("-")

def residual(form, test): # Residual helper function
    v = form.arguments()[0]
    return replace(form, {v: test})

sp_cell2   = {"mat_type": "matfree",
            "snes_type": "ksponly",
            "ksp_type": "cg",
            "pc_type": "jacobi",
            "pc_hypre_type": "pilut"}
sp_facet1    = {"mat_type": "matfree",
            "snes_type": "ksponly",
            "ksp_type": "cg",
            "pc_type": "jacobi",
            "pc_hypre_type": "pilut"}
    
def automatic_error_indicators(z_err, F):
    # 7. Compute cell and facet residuals R_T, R_\partialT
    dim = mesh.topological_dimension()
    cell = mesh.ufl_cell()
    
    variant = "integral" # Finite element type 

    # ---------------- Equation 4.6 to find cell residual Rcell -------------------------
    B = FunctionSpace(mesh, "B", dim+1, variant=variant) # Bubble function space
    bubbles = Function(B).assign(1) # Bubbles

    # Discontinuous function space of Rcell polynomials
    DG = FunctionSpace(mesh, "DG", residual_degree, variant=variant)

    uc = TrialFunction(DG)
    vc = TestFunction(DG)
    ac = inner(uc, bubbles*vc)*dx
    Lc = residual(F, bubbles*vc)

    Rcell = Function(DG, name="Rcell") # Rcell polynomial
    ndofs = DG.dim()
    #print("Computing Rcells ...")

    assemble(Lc)
    solve(ac == Lc, Rcell, solver_parameters=sp_cell2) # solve for Rcell polynonmial

    def both(u):
        return u("+") + u("-")

    # ---------------- Equation 4.8 to find facet residual Rfacet -------------------------
    FB = FunctionSpace(mesh, "FB", dim, variant=variant) # Cone function space
    cones = Function(FB).assign(1) # Cones

    el = BrokenElement(FiniteElement("FB", cell=cell, degree=residual_degree+dim, variant=variant))
    Q = FunctionSpace(mesh, el)
    Qtest = TestFunction(Q)
    Qtrial = TrialFunction(Q)
    Lf = residual(F, Qtest) - inner(Rcell, Qtest)*dx
    af = both(inner(Qtrial/cones, Qtest))*dS + inner(Qtrial/cones, Qtest)*ds

    Rhat = Function(Q)
    ndofs = Q.dim()
    #print("Computing Rfacets ...")
    solve(af == Lf, Rhat, solver_parameters=sp_facet1)
    Rfacet = Rhat/cones

    # 8. Compute error indicators eta_T 
    DG0 = FunctionSpace(mesh, "DG", degree=0)
    test = TestFunction(DG0)

    #print("Computing eta_T indicators ...")
    etaT = assemble(
        inner(inner(Rcell, z_err), test)*dx + 
        + inner(avg(inner(Rfacet, z_err)), both(test))*dS + 
        + inner(inner(Rfacet, z_err), test)*ds
    )
    return etaT

def run_exact(title, ncheck):
    """
    Compare computed eigenpairs with the exact analytic eigenfunctions.
    """
    print(title)
    print("\n# k  (m,n)   sigma_h     error_exact      error_predicted  "
          "effectivity1  sum(local errors)   effectivity2")

    used_p = set()
    rows = []

    for k, (lam_exact, m, n) in enumerate(exact_list[:ncheck], start=1):
        # --- exact reference eigenfunction in V_exact ---
        v_ref = Function(V_exact)
        v_ref.interpolate(exact_expr(m, n))
        l2_normalize(v_ref)

        # --- use exact eigenfunction as v_used ---
        v_used = v_ref

        # match a discrete eigenfunction in V
        idx_p, v_h = best_match(v_used, Vp_vecs, used_p)
        lamh = lam_h[idx_p]

        # interpolate exact function into V
        phi_h = Function(V)
        phi_h.interpolate(v_used)


        # continue with error analysis (same code as before)
        # ------------------------------------------------
        e = Function(V_exact); e.interpolate(v_used - v_h)
        sigma_h = 0.5 * assemble(inner(e, e) * dx)

        rhs = assemble(inner(grad(v_h), grad(v_used - phi_h)) * dx) \
              - lamh * assemble(inner(v_h, (v_used - phi_h)) * dx)

        error_exact = abs(lam_exact - lamh)
        denom = 1.0 - sigma_h
        error_pred = abs(rhs / denom) if abs(denom) > 1e-14 else float("nan")
        diff = abs(error_exact - error_pred)
        effectivity = error_pred / error_exact
        rows.append((k, m, n, error_exact, error_pred, diff))

        # -------------------- Error indicators --------------------
        # Manual
        n_f = FacetNormal(mesh)
        DG0 = FunctionSpace(mesh, "DG", degree=0)
        test = TestFunction(DG0)
        w = v_used - phi_h
        # eta_T = assemble(
        #             inner(div(grad(v_h)), w * test) * dx - 
        #         lamh * inner(v_h,w * test) * dx + 
        #         inner(0.5*jump(-grad(v_h), n_f), w * both(test)) * dS +
        #         inner(dot(-grad(v_h), n_f), w * test) * ds
        # )
        eta_T = assemble(
             inner(div(grad(v_h)), w * test) * dx +
            (-lamh * v_h * w) * test * dx + (jump(grad(v_h), n_f) * w) * 0.5*both(test) * dS
             + inner(dot(-grad(v_h), n_f), w * test) * ds
        )

        # Compute effectivity
        with eta_T.dat.vec_ro as v:         # read-only PETSc Vec
            eta_array = v.getArray().copy()   # NumPy array copy
        eta_array = np.abs(eta_array)
        total = eta_array.sum()
        total_normalised = total/denom
        eff2 = total_normalised/error_exact
        
        # Automatic
        v = TestFunction(V)
        form = inner(grad(v_h), grad(v)) * dx - lamh * inner(v_h, (v)) * dx
        eta_T_automatic = automatic_error_indicators(w, form)

        # Compute effectivity
        with eta_T_automatic.dat.vec_ro as v:         # read-only PETSc Vec
            eta_array_auto = v.getArray().copy()   # NumPy array copy
        eta_array_auto = np.abs(eta_array_auto)
        total_auto = eta_array_auto.sum()
        total_normalised_auto = total_auto/denom
        eff2_auto = total_normalised_auto/error_exact
        print(f"{k:2d}  ({m},{n})   {sigma_h: .6f}   {error_exact: .6e}     {error_pred: .6e}    {effectivity: .3f}     {total_normalised: .6e}      {eff2: .3f} ")
        print(f"Automatic:                                                            {total_normalised_auto: .6e}      {eff2_auto: .3f} ")

    return rows

rows_exact = run_exact("Using exact v:", ncheck=ncheck)
