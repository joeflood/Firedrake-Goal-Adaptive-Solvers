from firedrake import *
from netgen.occ import *
import csv
from eigen_solver_ctx import EigenSolverCtx
from typing import Callable, Any
from pathlib import Path
import numpy as np
from tsfc.ufl_utils import extract_firedrake_constants
import os
from functools import singledispatch
from firedrake.mg.ufl_utils import coarsen
from adaptive import AdaptiveMeshHierarchy
from adaptive_transfer_manager import AdaptiveTransferManager

# --- Complex helpers (two-real representation) ------------------------------

from dataclasses import dataclass

@dataclass
class CFun:
    r: Function  # real part
    i: Function  # imag part

def c_from_pair(vr: Function, vi: Function | None) -> CFun:
    if vi is None:
        vi = Function(vr.function_space()); vi.assign(0)
    return CFun(vr, vi)

def c_copy(u: CFun) -> CFun:
    ur = u.r.copy(deepcopy=True); ui = u.i.copy(deepcopy=True)
    return CFun(ur, ui)

def c_inner(u: CFun, v: CFun, Mform) -> complex:
    # Hermitian M-inner product: <u,v>_M = re + i*im
    def ip(a, b):  # <a,b>_M
        return float(assemble(replace_both(Mform, b, a)))
    re = ip(u.r, v.r) + ip(u.i, v.i)
    im = ip(u.r, v.i) - ip(u.i, v.r)
    return complex(re, im)

def c_norm(u: CFun, Mform) -> float:
    # ||u||_M = sqrt(<u,u>_M) ; imaginary part is ~0 numerically
    val = c_inner(u, u, Mform).real
    return (val if val > 0.0 else 0.0) ** 0.5

def c_normalize(u: CFun, Mform) -> CFun:
    n = c_norm(u, Mform)
    if n > 0:
        u.r.assign(u.r / n); u.i.assign(u.i / n)
    return u

def c_align(u: CFun, target: CFun, Mform) -> CFun:
    # phase-align u s.t. <target,u>_M is real & positive
    c = c_inner(target, u, Mform)
    if c == 0:
        return u
    phase = c.conjugate() / abs(c)
    ar, ai = phase.real, phase.imag
    ur = Function(u.r.function_space()); ui = Function(u.r.function_space())
    ur.assign(ar*u.r - ai*u.i)
    ui.assign(ai*u.r + ar*u.i)
    return CFun(ur, ui)

def c_diff(a: CFun, b: CFun) -> CFun:  # a - b
    ur = (a.r - b.r)
    ui = (a.i - b.i)
    return CFun(ur, ui)

import numpy as np

def gram_M(basis, Mform):
    """G_ij = <u_i, u_j>_M (Hermitian)."""
    m = len(basis)
    G = np.empty((m, m), dtype=complex)
    for i in range(m):
        for j in range(i, m):
            G[i, j] = c_inner(basis[i], basis[j], Mform)   # conjugate-linear in 1st arg
            if i != j:
                G[j, i] = np.conj(G[i, j])
    return 0.5*(G + G.conj().T)  # symmetrize numerically

def mix_basis(basis, C):
    """Return v_j = sum_i C[i,j]*basis[i] for j=0..r-1."""
    return [c_lincomb([C[i, j] for i in range(len(basis))], basis) for j in range(C.shape[1])]

def orthonormalize_M(basis, Mform, rtol=1e-12):
    """
    Build U' = U C so that (U')^* M U' = I.
    Fast path: Cholesky. Fallback: EVD with eigenvalue flooring for robustness.
    """
    if not basis:
        return []
    G = gram_M(basis, Mform)

    # Try Cholesky (SPD expected if vectors are independent)
    try:
        R = np.linalg.cholesky(G)                 # G = R^* R
        C = np.linalg.solve(R, np.eye(R.shape[0]))  # C = R^{-1}
        Uo = mix_basis(basis, C)
    except np.linalg.LinAlgError:
        # Robust fallback: eigen-decompose and renormalize
        w, V = np.linalg.eigh(G)                  # G = V diag(w) V^*
        wmax = max(w.max(), 1.0)
        w_floor = rtol * wmax
        w_clamped = np.clip(w.real, w_floor, None)  # floor tiny/neg eigenvalues
        C = V @ np.diag(1.0/np.sqrt(w_clamped)) @ V.conj().T
        Uo = mix_basis(basis, C)

    # (optional) sanity: make sure norms are 1
    # You can re-scale each vector if you want strictly unit diagonal.
    return Uo


class GoalAdaptiveEigenSolverComplex():
    '''
    Solves an eigenvalue problem adaptively. At the moment it only minimises λ-λ_h, but future editions could extend
    to allow goal functionals of the eigenfunctions J(u).
    The 'goal' in this context is the ability to target a particular eigenvalue.    
    '''

    def __init__(self, problem: LinearEigenproblem, target_eigenvalue: float, tolerance: float,  solver_parameters: dict,*, primal_solver_parameters = None, exact_solution = None):
        # User input vars
        self.problem = problem
        self.target = target_eigenvalue
        self.tolerance = tolerance
        self.sp_primal = primal_solver_parameters

        self.lam_exact = exact_solution
        self.solverctx = EigenSolverCtx(solver_parameters) # To store solver parameter data - Unnecessary, could remove in future.
        
        # Derived vars
        self.V = problem.output_space
        self.bcs = problem.bcs
        self.A = problem.A
        self.M = problem. M
        self.element = self.V.ufl_element()
        self.degree = self.element.degree()
        self.test = TestFunction(self.V)
        self.mesh = self.V.mesh()
        
        # Data storage and writing
        self.output_dir = Path(self.solverctx.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)  # ensures folder exists
        self.N_vec = []
        self.eta_vec = []
        self.etah_vec = []
        self.etaTsum_vec = []
        self.eff1_vec = []
        self.eff2_vec = []
        self.eff3_vec = []

    def solve_eigenproblems(self):
        s = self.solverctx
        def solve_eigs(Aform, Mform, bcs, nev, solver_parameters):
            prob = LinearEigenproblem(Aform, Mform, bcs=bcs, restrict=True)
            es = LinearEigensolver(prob, n_evals=nev, solver_parameters=solver_parameters)
            nconv = es.solve()
            lam, vecs = [], []
            for i in range(min(nconv, nev)):
                lam.append(es.eigenvalue(i))
                vr, vi = es.eigenfunction(i)
                u = c_from_pair(vr, vi)
                vecs.append(c_normalize(u, Mform))
            return lam, vecs

        def match_best_complex(target: CFun, candidates: list[CFun], Mform, lambdas=None):
            nt = c_norm(target, Mform)
            scores = []
            for i, w in enumerate(candidates):
                nw = c_norm(w, Mform)
                if nw == 0:
                    continue
                c = c_inner(target, w, Mform)
                corr = abs(c) / (nt * nw)
                scores.append((corr, i, c))

            if not scores:
                raise RuntimeError("No nonzero candidate matched.")

            scores.sort(key=lambda t: t[0], reverse=True)
            best_corr, best_i, best_c = scores[0]
            lam = lambdas[best_i] if lambdas is not None else None
            print(f" 1) idx={best_i:3d}, corr={best_corr:.6f}, <t,w>={best_c}, λ={lam}")
            for r, (corr, i, c) in enumerate(scores[1:6], start=2):
                lam_i = lambdas[i] if lambdas is not None else None
                print(f"{r:2d}) idx={i:3d}, corr={corr:.6f}, <t,w>={c}, λ={lam_i}")

            aligned = c_align(candidates[best_i], target, Mform)
            return (lam if lambdas is not None else best_i), aligned

        
        
        if s.self_adjoint == True:
            solver_parameters_target = {
                "eps_gen_hermitian": None,
                "eps_target": self.target
            }
        else:
            solver_parameters_target = {
                "eps_target": self.target
            }

        A = self.A
        M = self.M
        self.A_adj = adjoint(self.A)
        bcs=self.bcs
        NEV_SOLVE = self.solverctx.nev
        print("Primal DOFs: ", self.V.dim())
        self.N_vec.append(self.V.dim())

        dual_extra_degree = 1
        element = self.V.ufl_element()
        high_degree = self.degree + dual_extra_degree # By default use dual degree
        high_element = PMGPC.reconstruct_degree(element, high_degree)
        V_high = FunctionSpace(self.mesh, high_element)
        u_high = TrialFunction(V_high)
        v_high = TestFunction(V_high)

        A_high = replace_both(A, v_high, u_high)
        M_high = replace_both(M, v_high, u_high)
        bcs_high = reconstruct_bcs(bcs, V_high)
        A_adj_high  = replace_both(self.A_adj, v_high, u_high)

        lamh_prim_vec,  eigfunc_prim_vec   = solve_eigs(A,  M, bcs,    nev=NEV_SOLVE, solver_parameters=solver_parameters_target)
        self.u_h = eigfunc_prim_vec[0]
        self.lam_h = lamh_prim_vec[0]
        # Detect multiplicity 
        m = 1
        tolerance = 0.1
        for i in range(1, len(lamh_prim_vec)):
            if abs(lamh_prim_vec[i] - self.lam_h) <= tolerance:
                m += 1
        print("Eigenvalue multiplicity: ", m)
        self.cluster_basis_h = eigfunc_prim_vec[:m] 
        self.cluster_basis_h = orthonormalize_M(self.cluster_basis_h, self.M)
        

        lamh_prim_high_vec, eigfunc_prim_high_vec = solve_eigs(A_high, M_high, bcs_high, nev=NEV_SOLVE, solver_parameters=solver_parameters_target)
        
        
        print("Computed eigenvalue: ",self.lam_h)
        print("Matching eigenfunctions [primal in V_high]...")
        self.lam_p, self.u_p = match_best_complex(self.u_h, eigfunc_prim_high_vec , self.M, lamh_prim_high_vec)
        
        if s.self_adjoint == False:
            lamh_adj_vec, eigfunc_adj_vec = solve_eigs(self.A_adj, M, bcs, nev=NEV_SOLVE, solver_parameters=solver_parameters_target)
            lamh_adj_high_vec, eigfunc_adj_high_vec = solve_eigs(A_adj_high, M_high, bcs_high, nev=NEV_SOLVE, solver_parameters=solver_parameters_target)
            print("Matching eigenfunctions [dual in V]...")
            self.lamz_h, self.z_h = match_best_complex(self.u_h,eigfunc_adj_vec, self.M, lamh_adj_vec)
            print("Matching eigenfunctions [dual in V_high]...")
            self.lamz_p, self.z_p = match_best_complex(self.u_h,eigfunc_adj_high_vec, self.M, lamh_adj_high_vec)
        else:
            self.lamz_h =  self.lam_h
            self.z_h = self.u_h
            self.lamz_p = self.lam_p
            self.z_p = self.u_p

    def estimate_global_error(self):
        s = self.solverctx
        u_p, u_h = self.u_p, self.u_h
        z_p, z_h = self.z_p, self.z_h  # equals u_* if self-adjoint
        lam = self.lam_h
        lam_r, lam_i = float(np.real(lam)), float(np.imag(lam))


        # projection of u_p to V (separately for r/i)
        phi_r = Function(self.V); phi_r.interpolate(u_p.r)
        phi_i = Function(self.V); phi_i.interpolate(u_p.i)
        e        = CFun(u_p.r - phi_r, u_p.i - phi_i)

        e_sigma = c_diff(u_p, u_h)
        e_sigma_adj = c_diff(z_p, z_h)

        # sigma_h = 1/2 ||u - u_h||^2 (complex norm)
        #sigma_h = 0.5 * assemble((inner(e_sigma.r, e_sigma.r) + inner(e_sigma.i, e_sigma.i)) * dx)
        
        if s.self_adjoint:
            sigma_h = 0.5 * assemble((replace_both(self.M, e_sigma.r, e_sigma.r) + replace_both(self.M, e_sigma.i, e_sigma.i)))
            rhs = assemble(
                replace_both(self.A, u_h.r, e.r) - lam_r*replace_both(self.M, u_h.r, e.r)
            )
        else:
            sigma_h = 0.5 * assemble((replace_both(self.M, e_sigma.r, e_sigma_adj.r) + replace_both(self.M, e_sigma.i, e_sigma_adj.i)))
            # DWR split (real/imag): 0.5 * ( <A u_h - λ M u_h, e_adj> + <A^T z_h - λ M^T z_h, e> )
            e_adj = c_diff(z_p, z_h)
            rhs = 0.5*assemble(
                replace_both(self.A, u_h.r, e_adj.r) - lam_r*replace_both(self.M, u_h.r, e_adj.r) + lam_i*replace_both(self.M, u_h.i, e_adj.r)
                + replace_both(self.A, u_h.i, e_adj.i) - lam_r*replace_both(self.M, u_h.i, e_adj.i) - lam_i*replace_both(self.M, u_h.r, e_adj.i)
                + replace_both(self.A_adj, z_h.r, e.r) - lam_r*replace_both(self.M, z_h.r, e.r) + lam_i*replace_both(self.M, z_h.i, e.r)
                + replace_both(self.A_adj, z_h.i, e.i) - lam_r*replace_both(self.M, z_h.i, e.i) - lam_i*replace_both(self.M, z_h.r, e.i)
            )

        denom = 1.0 - sigma_h
        self.eta_h = abs(rhs / denom) if abs(denom) > 1e-14 else float("nan")
        self.etah_vec.append(self.eta_h)
        print(f"{'Predicted error:':45s}{':':8s}{self.eta_h:15.12f}")

        if self.lam_exact is not None:
            self.eta = abs(self.lam_exact - lam)  # ok if lam_exact is real or complex
            print(f"{'Exact error:':45s}{':':8s}{self.eta:15.12f}")
            self.eta_vec.append(self.eta)


    def automatic_error_indicators1(self):
        print("Computing local refinement indicators, η_K...")
        s = self.solverctx
        mesh = self.mesh
        u_h, z_p, z_h = self.u_h, self.z_p, self.z_h
        lam = self.lam_h
        lam_r, lam_i = float(np.real(lam)), float(np.imag(lam))

        # split residuals
        Fr = replace_trial(self.A, u_h.r) - lam_r*replace_trial(self.M, u_h.r) + lam_i*replace_trial(self.M, u_h.i)
        Fi = replace_trial(self.A, u_h.i) - lam_r*replace_trial(self.M, u_h.i) - lam_i*replace_trial(self.M, u_h.r)

        dim = mesh.topological_dimension()
        cell = mesh.ufl_cell()
        variant = "integral"
        deg_c = self.degree + s.cell_residual_extra_degree
        deg_f = self.degree + s.facet_residual_extra_degree

        B = FunctionSpace(mesh, "B", dim+1, variant=variant)
        bubbles = Function(B).assign(1)

        if self.V.value_shape == ():
            DG = FunctionSpace(self.mesh, "DG", deg_c, variant=variant)
        else:
            DG = TensorFunctionSpace(self.mesh, "DG", deg_c, variant=variant, shape=self.V.value_shape)

        uc = TrialFunction(DG); vc = TestFunction(DG)
        ac = inner(uc, bubbles*vc) * dx
        Lc_r = residual(Fr, bubbles*vc)
        Lc_i = residual(Fi, bubbles*vc)

        Rcell_r = Function(DG, name="Rcell_r")
        Rcell_i = Function(DG, name="Rcell_i")
        solve(ac == Lc_r, Rcell_r, solver_parameters=sp_cell2)
        solve(ac == Lc_i, Rcell_i, solver_parameters=sp_cell2)

        FB = FunctionSpace(mesh, "FB", dim, variant=variant)
        cones = Function(FB).assign(1)
        el = BrokenElement(FiniteElement("FB", cell=cell, degree=deg_f+dim, variant=variant))
        Q = FunctionSpace(mesh, el) if self.V.value_shape == () else TensorFunctionSpace(mesh, el, shape=self.V.value_shape)
        Qtest = TestFunction(Q); Qtrial = TrialFunction(Q)

        Lf_r = residual(Fr, Qtest) - inner(Rcell_r, Qtest)*dx
        Lf_i = residual(Fi, Qtest) - inner(Rcell_i, Qtest)*dx
        af   = both(inner(Qtrial/cones, Qtest))*dS + inner(Qtrial/cones, Qtest)*ds

        Rhat_r = Function(Q); Rhat_i = Function(Q)
        solve(af == Lf_r, Rhat_r, solver_parameters=sp_facet1)
        solve(af == Lf_i, Rhat_i, solver_parameters=sp_facet1)
        Rfacet_r = Rhat_r/cones
        Rfacet_i = Rhat_i/cones

        DG0 = FunctionSpace(mesh, "DG", degree=0)
        test = TestFunction(DG0)
        z_err = c_diff(z_p, z_h)

        self.etaT = assemble(
            (inner(Rcell_r,  z_err.r) + inner(Rcell_i,  z_err.i)) * test * dx
            + avg(inner(Rfacet_r, z_err.r) + inner(Rfacet_i, z_err.i)) * both(test) * dS
            + (inner(Rfacet_r,   z_err.r) + inner(Rfacet_i,   z_err.i)) * test * ds
        )

    def automatic_error_indicators(self):
        """
        Compute local refinement indicators η_K using BOTH primal and adjoint residuals,
        split into real/imag parts (so 4 local residual approximations: Fr_u, Fi_u, Fr_z, Fi_z).

        η_K ≈ 0.5 * [ <R_u, z_err> + <R_z, e_err> ] (cell + interior/boundary facet contributions)
        """
        print("Computing local refinement indicators, η_K (primal + adjoint)...")
        s = self.solverctx
        mesh = self.mesh

        # Ensure dual fields exist in self-adjoint mode
        if not hasattr(self, "z_h"):
            self.z_h = self.u_h
        if not hasattr(self, "z_p"):
            self.z_p = self.u_p

        u_h, z_h = self.u_h, self.z_h
        u_p, z_p = self.u_p, self.z_p

        lam = self.lam_h
        lam_r = float(np.real(lam))
        lam_i = float(np.imag(lam))

        # ----- Residual forms (split into real/imag) -----
        # Primal: (A - λ M) u = 0
        Fr_u = replace_trial(self.A, u_h.r) - lam_r*replace_trial(self.M, u_h.r) + lam_i*replace_trial(self.M, u_h.i)
        Fi_u = replace_trial(self.A, u_h.i) - lam_r*replace_trial(self.M, u_h.i) - lam_i*replace_trial(self.M, u_h.r)

        # Adjoint: (A^T - λ M^T) z = 0  (M is symmetric in our setting, so we reuse self.M)
        # self.A_adj should have been set in solve_eigenproblems()
        Fr_z = replace_trial(self.A_adj, z_h.r) - lam_r*replace_trial(self.M, z_h.r) + lam_i*replace_trial(self.M, z_h.i)
        Fi_z = replace_trial(self.A_adj, z_h.i) - lam_r*replace_trial(self.M, z_h.i) - lam_i*replace_trial(self.M, z_h.r)

        # ----- Spaces for local solves -----
        dim = mesh.topological_dimension()
        cell = mesh.ufl_cell()
        variant = "integral"
        deg_c = self.degree + s.cell_residual_extra_degree
        deg_f = self.degree + s.facet_residual_extra_degree

        B = FunctionSpace(mesh, "B", dim+1, variant=variant)
        bubbles = Function(B).assign(1)

        if self.V.value_shape == ():
            DG = FunctionSpace(mesh, "DG", deg_c, variant=variant)
        else:
            DG = TensorFunctionSpace(mesh, "DG", deg_c, variant=variant, shape=self.V.value_shape)

        uc = TrialFunction(DG)
        vc = TestFunction(DG)
        ac = inner(uc, bubbles*vc) * dx

        # ----- Cell residual solves (4 of them) -----
        Lc_r_u = residual(Fr_u, bubbles*vc)
        Lc_i_u = residual(Fi_u, bubbles*vc)
        Lc_r_z = residual(Fr_z, bubbles*vc)
        Lc_i_z = residual(Fi_z, bubbles*vc)

        Rcell_r_u = Function(DG, name="Rcell_r_u")
        Rcell_i_u = Function(DG, name="Rcell_i_u")
        Rcell_r_z = Function(DG, name="Rcell_r_z")
        Rcell_i_z = Function(DG, name="Rcell_i_z")

        solve(ac == Lc_r_u, Rcell_r_u, solver_parameters=sp_cell2)
        solve(ac == Lc_i_u, Rcell_i_u, solver_parameters=sp_cell2)
        solve(ac == Lc_r_z, Rcell_r_z, solver_parameters=sp_cell2)
        solve(ac == Lc_i_z, Rcell_i_z, solver_parameters=sp_cell2)

        # ----- Facet residual solves (4 of them) -----
        FB = FunctionSpace(mesh, "FB", dim, variant=variant)
        cones = Function(FB).assign(1)

        el = BrokenElement(FiniteElement("FB", cell=cell, degree=deg_f+dim, variant=variant))
        if self.V.value_shape == ():
            Q = FunctionSpace(mesh, el)
        else:
            Q = TensorFunctionSpace(mesh, el, shape=self.V.value_shape)

        Qtest = TestFunction(Q)
        Qtrial = TrialFunction(Q)

        Lf_r_u = residual(Fr_u, Qtest) - inner(Rcell_r_u, Qtest)*dx
        Lf_i_u = residual(Fi_u, Qtest) - inner(Rcell_i_u, Qtest)*dx
        Lf_r_z = residual(Fr_z, Qtest) - inner(Rcell_r_z, Qtest)*dx
        Lf_i_z = residual(Fi_z, Qtest) - inner(Rcell_i_z, Qtest)*dx

        af = both(inner(Qtrial/cones, Qtest))*dS + inner(Qtrial/cones, Qtest)*ds

        Rhat_r_u = Function(Q); Rhat_i_u = Function(Q)
        Rhat_r_z = Function(Q); Rhat_i_z = Function(Q)
        solve(af == Lf_r_u, Rhat_r_u, solver_parameters=sp_facet1)
        solve(af == Lf_i_u, Rhat_i_u, solver_parameters=sp_facet1)
        solve(af == Lf_r_z, Rhat_r_z, solver_parameters=sp_facet1)
        solve(af == Lf_i_z, Rhat_i_z, solver_parameters=sp_facet1)

        Rfacet_r_u = Rhat_r_u/cones
        Rfacet_i_u = Rhat_i_u/cones
        Rfacet_r_z = Rhat_r_z/cones
        Rfacet_i_z = Rhat_i_z/cones

        # ----- Error weights (align spaces onto V first) -----
        z_err = c_diff(z_p, z_h)
        u_err = c_diff(u_p, u_h)

        # ----- Assemble η_T (cell + facets), with 0.5 factor for symmetric DWR combination -----
        DG0 = FunctionSpace(mesh, "DG", degree=0)
        test = TestFunction(DG0)

        self.etaT = assemble(
            0.5 * (
                # primal residual weighted by z_err
                    (inner(Rcell_r_u,  z_err.r) + inner(Rcell_i_u,  z_err.i)) * test * dx
                    + avg(inner(Rfacet_r_u, z_err.r) + inner(Rfacet_i_u, z_err.i)) * both(test) * dS
                    + (inner(Rfacet_r_u,   z_err.r) + inner(Rfacet_i_u,   z_err.i)) * test * ds

                        # + adjoint residual weighted by e_err
                    + (inner(Rcell_r_z,  u_err.r) + inner(Rcell_i_z,  u_err.i)) * test * dx
                    + avg(inner(Rfacet_r_z, u_err.r) + inner(Rfacet_i_z, u_err.i)) * both(test) * dS
                    + (inner(Rfacet_r_z,   u_err.r) + inner(Rfacet_i_z,   u_err.i)) * test * ds
            )
        )



    def manual_error_indicators(self):
        ''' Currently only implemented for Poisson, but can be overriden. To adapt to other PDEs, replace the form of 
        self.etaT = assemble() to the symbolic form of the error indicators. This form is usually obtained by integrating 
        the weak form by parts (to recover the strong form) and redistributing facet fluxes.
        '''
        print("[MANUAL] Computing local refinement indicators (η_K)...")
        s = self.solverctx
        n = FacetNormal(self.mesh)
        DG0 = FunctionSpace(self.mesh, "DG", degree=0)
        test = TestFunction(DG0)
        self.etaT = assemble(
            inner(self.f + div(grad(self.u)), self.z_err * test) * dx +
            inner(0.5*jump(-grad(self.u), n), self.z_err * self.both(test)) * dS +
            inner(dot(-grad(self.u), n), self.z_err * test) * ds
        )

    def compute_efficiency(self):
        with self.etaT.dat.vec as evec:
            evec.abs()    
            self.etaT_array = evec.getArray()

        self.etaT_total = abs(np.sum(self.etaT_array))
        self.etaTsum_vec.append(self.etaT_total)
        print(f"{'Sum of refinement indicators':45s}{'Ση_K:':8s}{self.etaT_total:15.12f}")

        if self.lam_exact is not None:
            # Compute efficiency indices
            self.eff1 = self.eta_h/self.eta
            self.eff2 = self.etaT_total/self.eta
            print(f"{'Effectivity index 1':45s}{'η_h/η:':8s}{self.eff1:7.4f}")
            print(f"{'Effectivity index 2':45s}{'Ση_K/η:':8s}{self.eff2:7.4f}")
            self.eff1_vec.append(self.eff1)
            self.eff2_vec.append(self.eff2)
        else:
            self.eff3 = self.etaT_total/self.eta_h
            print(f"{'Effectivity index:':45s}{'Ση_K/η_h:':8s}{self.eff3:7.4f}")
            self.eff3_vec.append(self.eff3)

    def mark_cells(self):
        ''' Only Dorfler marking is implemented currently. Can be overridden if other marking strategies are desired. 
        '''
        s = self.solverctx
        # 9. Mark cells for refinement (Dorfler marking)
        sorted_indices = np.argsort(-self.etaT_array)
        sorted_etaT = self.etaT_array[sorted_indices]
        cumulative_sum = np.cumsum(sorted_etaT)
        threshold = s.dorfler_alpha * self.etaT_total
        M = np.searchsorted(cumulative_sum, threshold) + 1
        marked_cells = sorted_indices[:M]

        markers_space = FunctionSpace(self.mesh, "DG", 0)
        self.markers = Function(markers_space)
        with self.markers.dat.vec as mv:
            marr = mv.getArray()
            marr[:] = 0
            marr[marked_cells] = 1

    def uniform_refine(self):
        # Uniform marking for comparison tests
        markers_space = FunctionSpace(self.mesh, "DG", 0)
        self.markers = Function(markers_space)
        self.markers.assign(1)        

    def refine_mesh(self):
        
        def _rebind_form_to_mesh(form, V_new, mesh_new):
            """Put form on V_new / mesh_new. Handles the common '... * dx' case."""
            args = form.arguments()
            if len(args) == 2:
                v_old, u_old = args
                v_new, u_new = TestFunction(V_new), TrialFunction(V_new)
                F = replace(form, {v_old: v_new, u_old: u_new})
            elif len(args) == 1:
                v_old, = args
                v_new = TestFunction(V_new)
                F = replace(form, {v_old: v_new})
            else:
                return form  # 0-arg form

            # rebuild with dx on the new mesh (cell terms)
            integrand = sum(itg.integrand() for itg in F.integrals() if itg.integral_type() == "cell")
            return integrand * dx(domain=mesh_new)

        def _reconstruct_bcs_on(V_new, bcs_old):
            """Rebuild DirichletBCs on V_new (interpolating values if needed)."""
            new_bcs = []
            for bc in bcs_old:
                Vt = V_new
                for idx in getattr(bc, "_indices", ()):
                    Vt = Vt.sub(idx)
                g_old = bc._original_arg
                if isinstance(g_old, firedrake.Function):
                    g_new = firedrake.Function(Vt)
                    g_new.interpolate(g_old)
                else:
                    g_new = g_old
                new_bcs.append(DirichletBC(Vt, g_new, bc.sub_domain))
            return new_bcs
        
        print("Refining mesh ...")
        new_mesh = self.mesh.refine_marked_elements(self.markers)
        print("Transferring problem to new mesh ...")
        V_new = FunctionSpace(new_mesh, self.V.ufl_element())
        # rebuild forms on the new mesh
        A_new = _rebind_form_to_mesh(self.A, V_new, new_mesh)
        M_new = _rebind_form_to_mesh(self.M, V_new, new_mesh)

        # rebuild BCs
        bcs_new = _reconstruct_bcs_on(V_new, self.bcs)

        # replace problem & cached handles
        self.problem = LinearEigenproblem(A_new, M_new, bcs=bcs_new)
        self.mesh = new_mesh
        self.V, self.A, self.M, self.bcs = V_new, A_new, M_new, bcs_new
        self.mesh = new_mesh
       
    def write_data(self):
        s = self.solverctx
        # Write to file
        if s.results_file_name is None:
            file_path = self.output_dir / "results.csv"
        else:
            file_path = self.output_dir / s.results_file_name
        rows = list(zip(self.N_vec, self.Ndual_vec, self.eta_vec, self.etah_vec, self.etaTsum_vec, self.eff1_vec, self.eff2_vec))
        headers = ("N", "Ndual", "eta", "eta_h", "sum_eta_T", "eff1", "eff2")
        with open(file_path, "w", newline="") as file:
            w = csv.writer(file)
            w.writerow(headers)
            w.writerows(rows)
            jump

    def append_data(self, it):
        s = self.solverctx
        if s.results_file_name is None:
            file_path = self.output_dir / "results.csv"
        else:
            file_path = self.output_dir / s.results_file_name
        if self.lam_exact is None and self.solverctx.uniform_refinement == False:
            headers = ("iteration", "N", "Ndual", "eta_h", "sum_eta_T")
            row = (
                it,
                self.N_vec[-1], self.etah_vec[-1], self.etaTsum_vec[-1]
            )
        elif self.solverctx.uniform_refinement == True:
            headers = ("iteration", "N", "Ndual", "eta", "eta_h")
            row = (
                it,
                self.N_vec[-1], self.eta_vec[-1], self.etah_vec[-1]
            )
        else:
            headers = ("iteration", "N", "Ndual", "eta", "eta_h", "sum_eta_T", "eff1", "eff2")
            row = (
                it,
                self.N_vec[-1], self.eta_vec[-1], self.etah_vec[-1], self.etaTsum_vec[-1], self.eff1_vec[-1], self.eff2_vec[-1]
            )
        
        file_exists = os.path.exists(file_path)

        if it == 0:
            with open(file_path, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(headers)
                writer.writerow(row)
        else:
            with open(file_path, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(row)
    
    def write_mesh(self,it):
        s = self.solverctx
        should_write = False
        if s.write_mesh == "all":
            should_write = True
        elif s.write_mesh == "first_and_last":
                if it == 0 or it == s.max_iterations:
                    should_write = True
        elif s.write_mesh == "by_iteration":
                # Case A: user gave specific iterations (list/tuple/set)
                if getattr(s, "write_mesh_iteration_vector", None) is not None:
                    # allow any iterable; convert to set for O(1) lookup
                    targets = set(s.write_iteration_vector)
                    should_write = it in targets
                # Case B: otherwise use interval (positive int)
                elif getattr(s, "write_mesh_iteration_interval", None) is not None:
                    interval = int(s.write_iteration_interval)
                    if interval <= 0:
                        raise ValueError("write_mesh_iteration_interval must be a positive integer")
                    should_write = (it % interval == 0)  # includes it=0
        if should_write:
            print("Writing mesh ...")
            VTKFile(self.output_dir / f"mesh{it}.pvd").write(self.mesh)

    def write_solution(self,it):
        s = self.solverctx
        should_write = False
        if s.write_solution == "all":
            should_write = True
        elif s.write_solution == "first_and_last":
                if it == 0 or it == s.max_iterations:
                    should_write = True
        elif s.write_solution == "by_iteration":
                # Case A: user gave specific iterations (list/tuple/set)
                if getattr(s, "write_solution_iteration_vector", None) is not None:
                    # allow any iterable; convert to set for O(1) lookup
                    targets = set(s.write_iteration_vector)
                    should_write = it in targets
                # Case B: otherwise use interval (positive int)
                elif getattr(s, "write_solution_iteration_interval", None) is not None:
                    interval = int(s.write_iteration_interval)
                    if interval <= 0:
                        raise ValueError("write_solution_iteration_interval must be a positive integer")
                    should_write = (it % interval == 0)  # includes it=0
        if should_write:
            print("Writing (primal) solution (real & imag)...")
            VTKFile(self.output_dir / f"solution_{it}.pvd").write(*self.u_h.r.subfunctions,
            *self.u_h.i.subfunctions)

            # --- pull out the velocity block (index 0) from u_h.r / u_h.i ---
            # if not mixed, this just returns the whole function
            vel_r = self.u_h.r.subfunctions[0] if getattr(self.u_h.r, "subfunctions", ()) else self.u_h.r

            if s.self_adjoint == True:
                mag_expr = sqrt(inner(vel_r, vel_r))
            else:
                vel_i = self.u_h.i.subfunctions[0] if getattr(self.u_h.i, "subfunctions", ()) else self.u_h.i
                mag_expr = sqrt(inner(vel_r, vel_r) + inner(vel_i, vel_i))
            
            # --- build a scalar space for the magnitude ---
            Vm = FunctionSpace(self.mesh, "CG", 2)
            vel_mag = Function(Vm, name="|u_h| (vel)")
            vel_mag.interpolate(mag_expr)

            # --- assemble outputs: velocity real, imag, magnitude
            outs = [vel_r, vel_mag]

            # (optional) also dump pressure real/imag if you want:
            if getattr(self.u_h.r, "subfunctions", ()) and len(self.u_h.r.subfunctions) > 1:
                p_r = self.u_h.r.subfunctions[1]
                p_i = self.u_h.i.subfunctions[1]
                outs.extend([p_r, p_i])

            VTKFile(self.output_dir / f"solution_{it}.pvd").write(*outs)

    def solve(self):
        s = self.solverctx

        for it in range(s.max_iterations):
            print(f"---------------------------- [MESH LEVEL {it}] ----------------------------")
            self.write_mesh(it)
            self.solve_eigenproblems()
            self.write_solution(it)
            self.estimate_global_error()
            if self.eta_h < self.tolerance:
                print("Error estimate below tolerance, finished.")
                break
            if it == s.max_iterations -1:
                print(f"Maximum iteration ({s.max_iterations}) reached. Exiting.")
                break
            if s.uniform_refinement == True:
                print("Refining uniformly")
                self.uniform_refine()
            else:
                if s.manual_indicators == True:
                    self.manual_error_indicators()
                else:
                    self.automatic_error_indicators()
                self.compute_efficiency()
                self.mark_cells()
            if s.write_at_iteration == True:
                print("Appending data ...")
                self.append_data(it)
            self.refine_mesh()

        if s.write_at_iteration == False:
            print("Writing data ...")
            self.write_data()
    
    def both(self, u):
        return u("+") + u("-")


def getlabels(mesh): # Doesn't seem to work in 2D ?
    ngmesh = mesh.netgen_mesh
    names = ngmesh.GetRegionNames(codim=2)
    print(names)
    names_to_labels = {}
    for l in names:
        names_to_labels[l] = tuple(i+1 for i, name in enumerate(names) if name == l)
        print(names_to_labels[l])
    return names_to_labels


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

def residual(F, test): # Residual helper function
    v = F.arguments()[0]
    return replace(F, {v: test})


# For transferring to new mesh :

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


# Residual-based indicators
def both(u):
    return u("+") + u("-")

sp_cell2   = {"mat_type": "matfree", "snes_type": "ksponly", "ksp_type": "cg", "pc_type": "jacobi", "pc_hypre_type": "pilut"}
sp_facet1  = {"mat_type": "matfree", "snes_type": "ksponly", "ksp_type": "cg", "pc_type": "jacobi", "pc_hypre_type": "pilut"}


from ufl.duals import is_dual
from firedrake.dmhooks import get_transfer_manager, get_appctx

@singledispatch
def refine(expr, self, coefficient_mapping=None):
    return coarsen(expr, self, coefficient_mapping=coefficient_mapping)  # fallback to original

@refine.register(firedrake.Cofunction)
@refine.register(firedrake.Function)
def refine_function(expr, self, coefficient_mapping=None):
    if coefficient_mapping is None:
        coefficient_mapping = {}
    new = coefficient_mapping.get(expr)
    if new is None:
        Vf = expr.function_space()
        Vc = self(Vf, self)
        new = firedrake.Function(Vc, name=f"coarse_{expr.name()}")
        manager = get_transfer_manager(Vf.dm)
        if is_dual(expr):
            print("is_dual called! (function)")
            manager.restrict(expr, new)
        else:
            print("is_dual not called! (function)")
            manager.prolong(expr, new)
        coefficient_mapping[expr] = new
    return new

@refine.register(firedrake.LinearEigenproblem)
def refine_eigenproblem(problem, self, coefficient_mapping=None):
    if hasattr(problem, "_coarse"):
        return problem._coarse

    def inject_on_restrict(fine, restriction, rscale, injection, coarse):
        manager = get_transfer_manager(fine)
        cctx = get_appctx(coarse)
        cmapping = cctx._coefficient_mapping
        if cmapping is None:
            return
        for c in cmapping:
            if is_dual(c):
                print("is_dual called! (eigenproblem)")
                manager.restrict(c, cmapping[c])
            else:
                print("is_dual not called! (eigenproblem)")
                manager.prolong(c, cmapping[c])
        # Apply bcs
        if cctx.pre_apply_bcs:
            for bc in cctx._problem.dirichlet_bcs():
                bc.apply(cctx._x)

    dm = problem.output_space.dm
    if not dm.getAttr("_coarsen_hook"):
        # The hook is persistent and cumulative, but also problem-independent.
        # Therefore, we are only adding it once.
        dm.addCoarsenHook(None, inject_on_restrict)
        dm.setAttr("_coarsen_hook", True)

    if coefficient_mapping is None:
        coefficient_mapping = {}

    bcs = [self(bc, self, coefficient_mapping=coefficient_mapping) for bc in problem.bcs]
    A = self(problem.A, self, coefficient_mapping=coefficient_mapping)
    M = self(problem.M, self, coefficient_mapping=coefficient_mapping)

    fine = problem
    problem = firedrake.LinearEigenproblem(A, M, bcs=bcs)
    fine._coarse = problem
    return problem


@refine.register(firedrake.DirichletBC)
def refine_bc(bc, self, coefficient_mapping=None):
    V   = self(bc.function_space(), self, coefficient_mapping=coefficient_mapping)
    val = self(bc.function_arg,   self, coefficient_mapping=coefficient_mapping)
    if isinstance(val, firedrake.Function) and val.function_space() != V:
        valr = firedrake.Function(V, name=getattr(val, "name", lambda: "bc_val")())
        print("Trying to reassign g to new function space ...")
        try:
            valr.interpolate(val)
        except Exception:
            valr.assign(0)  # fallback; homogeneous is typical
        val = valr
    return type(bc)(V, val, bc.sub_domain)