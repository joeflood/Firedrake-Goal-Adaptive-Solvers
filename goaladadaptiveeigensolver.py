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


class GoalAdaptiveEigenSolver():
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
                vecs.append(l2_normalize(vr))
            return lam, vecs
        
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
        lamh_prim_high_vec, eigfunc_prim_high_vec = solve_eigs(A_high, M_high, bcs_high, nev=NEV_SOLVE, solver_parameters=solver_parameters_target)
        
        self.u_h = eigfunc_prim_vec[0]
        self.lam_h = lamh_prim_vec[0]
        print("Computed eigenvalue: ",self.lam_h)
        print("Matching eigenfunctions [primal in V_high]...")
        self.lam_p, self.u_p = match_best(self.u_h, eigfunc_prim_high_vec , lamh_prim_high_vec)
        
        if s.self_adjoint == False:
            lamh_adj_vec, eigfunc_adj_vec = solve_eigs(self.A_adj, M, bcs, nev=NEV_SOLVE, solver_parameters=solver_parameters_target)
            lamh_adj_high_vec, eigfunc_adj_high_vec = solve_eigs(A_adj_high, M_high, bcs_high, nev=NEV_SOLVE, solver_parameters=solver_parameters_target)
            print("Matching eigenfunctions [dual in V]...")
            self.lamz_h, self.z_h = match_best(self.u_h,eigfunc_adj_vec,lamh_adj_vec)
            print("Matching eigenfunctions [dual in V_high]...")
            self.lamz_p, self.z_p = match_best(self.u_h,eigfunc_adj_high_vec,lamh_adj_high_vec)

    def estimate_global_error(self):
        s = self.solverctx
        u_p = self.u_p
        u_h = self.u_h
        lam_h = self.lam_h
        z_p = self.z_p
        z_h = self.z_h
        A = self.A
        M = self.M
        A_adj = self.A_adj

        phi_h = Function(self.V)
        phi_h.interpolate(u_p)
        e = u_p - phi_h
        e_sigma = u_p - u_h

        # sigma_h = 1/2 ||u - u_h||^2
        if s.self_adjoint == True:
            sigma_h = 0.5 * assemble(inner(e_sigma, e_sigma) * dx)
            rhs = assemble( action(replace_trial(A,u_h),e) - lam_h * replace_test(replace_trial(M,u_h),e)
                            )
        else:
            e_sigma_adj = z_p - z_h
            sigma_h = 0.5 * assemble(inner(e_sigma, e_sigma_adj) * dx)
            e_adj = z_p - z_h
            rhs = 0.5* assemble( replace_both(A,u_h,e_adj) - lam_h * replace_both(M,u_h,e_adj)
                    + replace_both(A_adj,z_h,e) - lam_h * replace_both(M,z_h,e)
                    )
    
        denom = 1.0 - sigma_h
        self.eta_h = abs(rhs / denom) if abs(denom) > 1e-14 else float("nan")
        self.etah_vec.append(self.eta_h)

        print(f"{'Predicted error:':45s}{':':8s}{self.eta_h:15.12f}")

        if self.lam_exact is not None:
            self.eta = abs(self.lam_exact - lam_h)
            print(f"{'Exact error:':45s}{':':8s}{self.eta:15.12f}")
        self.eta_vec.append(self.eta)

    def automatic_error_indicators(self):
        print("Computing local refinement indicators, η_K...")
        # cell residual
        s = self.solverctx
        mesh = self.mesh
        u_p = self.u_p
        u_h = self.u_h
        lam_h = self.lam_h
        z_p = self.z_p
        z_h = self.z_h
        z_err = z_p - z_h

        F = replace_trial(self.A,u_h) - lam_h * replace_trial(self.M, u_h)
        dim = mesh.topological_dimension()
        cell = mesh.ufl_cell()
        variant = "integral"
        cell_residual_degree = self.degree + s.cell_residual_extra_degree
        facet_residuaL_degree = self.degree + s.facet_residual_extra_degree

        B  = FunctionSpace(mesh, "B", dim+1, variant=variant)
        bubbles = Function(B).assign(1)

        #DG = FunctionSpace(mesh, "DG", 1, variant=variant)
        print("Shape of V: ", self.V.value_shape)
        if self.V.value_shape == ():
            DG = FunctionSpace(self.mesh, "DG", cell_residual_degree, variant=variant)
        else:
            DG = TensorFunctionSpace(self.mesh, "DG", cell_residual_degree, variant=variant, shape=self.V.value_shape)
        uc = TrialFunction(DG); vc = TestFunction(DG)
        ac = inner(uc, bubbles*vc)*dx
        print("Shape of F.u", F.arguments()[0].function_space().value_shape)
        print("Shape of bubbles*vc: ", DG.value_shape)
        Lc = residual(F, bubbles*vc)

        Rcell = Function(DG, name="Rcell")
        solve(ac == Lc, Rcell, solver_parameters=sp_cell2)

        # facet residual
        FB = FunctionSpace(mesh, "FB", dim, variant=variant)
        cones = Function(FB).assign(1)

        el = BrokenElement(FiniteElement("FB", cell=cell, degree=facet_residuaL_degree+dim, variant=variant))
        if self.V.value_shape == ():
            Q = FunctionSpace(self.mesh, el)
        else: 
            Q = TensorFunctionSpace(self.mesh, el, shape=self.V.value_shape)

        Qtest = TestFunction(Q); Qtrial = TrialFunction(Q)
        Lf = residual(F, Qtest) - inner(Rcell, Qtest)*dx
        af = both(inner(Qtrial/cones, Qtest))*dS + inner(Qtrial/cones, Qtest)*ds

        Rhat = Function(Q)
        solve(af == Lf, Rhat, solver_parameters=sp_facet1)
        Rfacet = Rhat/cones

        # indicators
        DG0 = FunctionSpace(mesh, "DG", degree=0)
        test = TestFunction(DG0)
        self.etaT = assemble(
            inner(inner(Rcell, z_err), test)*dx
        + inner(avg(inner(Rfacet, z_err)), both(test))*dS
        + inner(inner(Rfacet, z_err), test)*ds
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
        print("Refining mesh ...")
        new_mesh = self.mesh.refine_marked_elements(self.markers)
        print("Transferring problem to new mesh ...")
        amh = AdaptiveMeshHierarchy([self.mesh])
        atm = AdaptiveTransferManager()
        amh.add_mesh(new_mesh)
        coef_map = {}
        self.problem = refine(self.problem, refine, coefficient_mapping=coef_map)

        self.A = self.problem.A
        self.M = self.problem.M
        self.bcs = self.problem.bcs
        self.V = self.problem.output_space
        #self.problem = NonlinearVariationalProblem(self.F,self.u,self.bcs)
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
            print("Writing (primal) solution ...")
            VTKFile(self.output_dir / f"solution_{it}.pvd").write(*self.u_h.subfunctions)
            
            #if 
            #print("Writing (primal) magnitude ...")

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
        vh = Function(Vspace); vh.assign(vr)
        vecs.append(l2_normalize(vh))
    return lam, vecs

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


def l2_normalize(f):
            nrm = assemble(inner(f, f)*dx)**0.5
            if nrm > 0:
                f.assign(f/nrm)
            return f

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
            manager.restrict(expr, new)
        else:
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
                manager.restrict(c, cmapping[c])
            else:
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