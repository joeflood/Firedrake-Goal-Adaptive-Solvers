from firedrake import *
from netgen.occ import *
import csv
from solver_ctx import SolverCtx
from typing import Callable, Any
from pathlib import Path
import numpy as np
from tsfc.ufl_utils import extract_firedrake_constants
import os
from functools import singledispatch
from firedrake.mg.ufl_utils import coarsen
from adaptive import AdaptiveMeshHierarchy
from adaptive_transfer_manager import AdaptiveTransferManager

class GoalAdaptiveNonlinearVariationalSolver():
    '''
    Solves a nonlinear variational problem to minimise the error in a user-specified goal functional.
    We do this by adaptively refining the mesh based on the solution to a dual problem - which links the goal functional 
    to the PDE.     
    '''

    def __init__(self, problem: NonlinearVariationalProblem, goal_functional, tolerance: float,  solver_parameters: dict,
                 *, primal_solver_parameters = None, dual_solver_parameters = None, exact_solution = None, exact_goal = None,
                 nullspace = None):
        # User input vars
        self.problem = problem
        self.J = goal_functional
        self.tolerance = tolerance
        self.sp_primal = primal_solver_parameters
        self.sp_dual = dual_solver_parameters
        self.u_exact = exact_solution
        self.goal_exact = exact_goal
        self.solverctx = SolverCtx(solver_parameters) # To store solver parameter data - Unnecessary, could remove in future.
        # Derived vars
        self.V = problem.u.function_space()
        self.u = problem.u
        self.bcs = problem.bcs
        self.F = problem.F
        self.element = self.V.ufl_element()
        self.degree = self.element.degree()
        self.test = TestFunction(self.V)
        self.mesh = self.V.mesh()
        self.nullspace = nullspace

        # Data storage and writing
        self.output_dir = Path(self.solverctx.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)  # ensures folder exists
        self.N_vec = []
        self.Ndual_vec = []
        self.eta_vec = []
        self.etah_vec = []
        self.etaTsum_vec = []
        self.eff1_vec = []
        self.eff2_vec = []
        self.eff3_vec = []

    def solve_primal(self):
        s = self.solverctx
        ndofs = self.V.dim()
        self.N_vec.append(ndofs)
        def solve_uh():
            if self.sp_primal is None:
                print(f"Solving primal (degree: {self.degree}, dofs: {ndofs}) [Method: Default nonlinear solve] ...")
                NonlinearVariationalSolver(self.problem).solve()
            else:
                print(f"Solving primal (degree: {self.degree}, dofs: {ndofs}) [Method: User defined] ...")
                NonlinearVariationalSolver(self.problem, solver_parameters=self.sp_primal).solve()
        
        if s.use_adjoint_residual == True:
            # Now solve in higher space
            high_degree = self.degree + s.dual_extra_degree # By default use dual degree
            high_element = PMGPC.reconstruct_degree(self.element, high_degree)
            Vhigh = FunctionSpace(self.mesh, high_element)
            test = TestFunction(Vhigh) # Dual test function
            self.u_high = Function(Vhigh) # Dual soluton
            (v_old,) = self.F.arguments()
            v_high = TestFunction(Vhigh)
            F_high = ufl.replace(self.F, {v_old: v_high, self.u: self.u_high})
            bcs_high = reconstruct_bcs(self.bcs, Vhigh)
            self.problem_high = NonlinearVariationalProblem(F_high, self.u_high, bcs_high)
            
            if self.sp_primal is None:
                print(f"Solving primal in higher space for error estimate (degree: {high_degree}, dofs: {Vhigh.dim()}) [Method: Default nonlinear solve] ...")
                NonlinearVariationalSolver(self.problem_high).solve()
            else:
                print(f"Solving primal in higher space for error estimate (degree: {high_degree}, dofs: {Vhigh.dim()}) [Method: User defined] ...")
                NonlinearVariationalSolver(self.problem_high, solver_parameters=self.sp_primal).solve()
            
            if s.primal_low_method == "solve":
                solve_uh()
            elif s.primal_low_method == "project":
                self.u.project(self.u_high)
            else:
                self.u.interpolate(self.u_high) # Default - but gives bad results?
            self.u_err = self.u_high - self.u
        else:
            solve_uh()

    def solve_dual(self):
        s = self.solverctx

        dual_degree = self.degree + s.dual_extra_degree
        dual_element = PMGPC.reconstruct_degree(self.element, dual_degree)
        Vdual = FunctionSpace(self.mesh, dual_element)
        vtest = TestFunction(Vdual) # Dual test function
        self.z = Function(Vdual) # Dual soluton

        ndofs_dual = Vdual.dim()
        self.Ndual_vec.append(ndofs_dual)    

        self.G = ( action(adjoint(derivative(self.F, self.u, TrialFunction(Vdual))), self.z) 
             - derivative(self.J, self.u, vtest) )
        
        bcs_dual  = [bc.reconstruct(V=Vdual, indices=bc._indices, g=0) for bc in self.bcs]
                 
        if self.sp_dual is None:
            print(f"Solving dual (degree: {dual_degree}, dofs: {ndofs_dual}) [Method: Default nonlinear solve] ...")
            solve(self.G == 0, self.z, bcs_dual)
            
        else:
            print(f"Solving dual (degree: {dual_degree}, dofs: {ndofs_dual}) [Method: User defined] ...")
            solve(self.G == 0, self.z, bcs_dual, solver_parameters=self.sp_dual)
        
        # zlo
        self.z_lo = Function(self.V) # Dual soluton
        if s.dual_low_method == "solve":
            ndofs = self.V.dim()
            test = TestFunction(self.V)
            G_lo = ( action(adjoint(derivative(self.F, self.u, TrialFunction(self.V))), self.z_lo) 
                - derivative(self.J, self.u, test) )
            
            bcs_dual_low  = [bc.reconstruct(V=self.V, indices=bc._indices, g=0) for bc in self.bcs]
                    
            if self.sp_dual is None:
                print(f"Solving dual in V (degree: {self.degree}, dofs: {ndofs}) [Method: Default nonlinear solve] ...")
                solve(G_lo == 0, self.z_lo, bcs_dual_low)
                
            else:
                print(f"Solving dual in V (degree: {self.degree}, dofs: {ndofs}) [Method: User defined] ...")
                solve(G_lo == 0, self.z_lo, bcs_dual_low, solver_parameters=self.sp_dual)
        elif s.dual_low_method == "project":
            self.z_lo.project(self.z)
        else:
            self.z_lo.interpolate(self.z) #Default method
        
        self.z_err = self.z - self.z_lo

    def compute_etah(self):
        s = self.solverctx
        # Compute error estimate F(z)
        if s.use_adjoint_residual == True:
            primal_err = abs(assemble(self.residual(self.F, self.z_err)))

            G = ( action(adjoint(derivative(self.F, self.u, TrialFunction(self.V))), self.z_lo) 
             - derivative(self.J, self.u, TestFunction(self.V)) )
            dual_err = abs(assemble(self.residual(G, self.u_err)))
            self.eta_h = abs(0.5* primal_err + 0.5*dual_err)
        else:
            self.eta_h = abs(assemble(self.residual(self.F, self.z_err)))

        self.etah_vec.append(self.eta_h)

        self.Juh = assemble(self.J)
        print(f"{'Computed goal':45s}{'J(uh):':8s}{self.Juh:15.12f}")

        if self.u_exact is not None:
            quad_opts = {"quadrature_degree": 20}
            def as_mixed(exprs):
                return as_vector([e[idx] for e in exprs for idx in np.ndindex(e.ufl_shape)])
            if type(self.u_exact) == list or type(self.u_exact) == tuple:
                self.Ju = assemble(replace(self.J, {self.u: as_mixed(self.u_exact)}), form_compiler_parameters=quad_opts)
            else:
                self.Ju = assemble(replace(self.J, {self.u: self.u_exact}), form_compiler_parameters=quad_opts)
        elif self.goal_exact is not None:
            self.Ju = self.goal_exact
        
        if self.u_exact is not None or self.goal_exact is not None:
            self.eta = abs(self.Juh - self.Ju)
            self.eta_vec.append(self.eta)
            print(f"{'Exact goal':45s}{'J(u):':8s}{self.Ju:15.12f}")
            print(f"{'True error, |J(u) - J(u_h)|':45s}{'η:':8s}{self.eta:15.12f}")
        
        if s.use_adjoint_residual == True:
            print(f"{'Primal error, |ρ(u_h;z-z_h)|:':45s}{'η_pri:':8s}{primal_err:15.12f}")
            print(f"{'Dual error, |ρ*(z_h;u-u_h)|:':45s}{'η_adj:':8s}{dual_err:15.12f}")
            print(f"{'Difference':35s}{'|η_pri - η_adj|:':18s}{abs(primal_err-dual_err):19.12e}")
            print(f"{'Predicted error, 0.5|ρ+ρ*|':45s}{'η_h:':8s}{self.eta_h:15.12f}")
        else:
             print(f"{'Predicted error, |ρ(u_h;z-z_h)|':45s}{'η_h:':8s}{self.eta_h:15.12f}")         

    def automatic_error_indicators(self):
        print("Computing local refinement indicators, η_K...")
        # 7. Compute cell and facet residuals R_T, R_\partialT       
        s = self.solverctx
        dim = self.mesh.topological_dimension()
        cell = self.mesh.ufl_cell()
        variant = "integral" # Finite element type 
        cell_residual_degree = self.degree + s.cell_residual_extra_degree
        facet_residuaL_degree = self.degree + s.facet_residual_extra_degree
        # ------------------------------- Primal residual -------------------------------
        # Rcell
        B = FunctionSpace(self.mesh, "B", dim+1, variant=variant) # Bubble function space
        bubbles = Function(B).assign(1) # Bubbles
        # Discontinuous function space of Rcell polynomials
        if self.V.value_shape == ():
            DG = FunctionSpace(self.mesh, "DG", cell_residual_degree, variant=variant)
        else:
            DG = TensorFunctionSpace(self.mesh, "DG", cell_residual_degree, variant=variant, shape=self.V.value_shape)
        uc = TrialFunction(DG)
        vc = TestFunction(DG)
        ac = inner(uc, bubbles*vc)*dx
        Lc = self.residual(self.F, bubbles*vc)
        Rcell = Function(DG, name="Rcell") # Rcell polynomial
        assemble(Lc)
        solve(ac == Lc, Rcell, solver_parameters=s.sp_cell) # solve for Rcell polynonmial

        # Rfacet
        FB = FunctionSpace(self.mesh, "FB", dim, variant=variant) # Cone function space
        cones = Function(FB).assign(1) # Cones
        # Broken discontinuous function space of facet polynomials
        el = BrokenElement(FiniteElement("FB", cell=cell, degree=facet_residuaL_degree+dim, variant=variant))
        if self.V.value_shape == ():
            Q = FunctionSpace(self.mesh, el)
        else: 
            Q = TensorFunctionSpace(self.mesh, el, shape=self.V.value_shape)
        Qtest = TestFunction(Q)
        Qtrial = TrialFunction(Q)
        Lf = self.residual(self.F, Qtest) - inner(Rcell, Qtest)*dx
        af = self.both(inner(Qtrial/cones, Qtest))*dS + inner(Qtrial/cones, Qtest)*ds
        Rhat = Function(Q)
        solve(af == Lf, Rhat, solver_parameters=s.sp_facet)
        Rfacet = Rhat/cones
        
        # Primal error indicators
        DG0 = FunctionSpace(self.mesh, "DG", degree=0)
        test = TestFunction(DG0)

        eta_primal = assemble(
            inner(inner(Rcell, self.z_err), test)*dx + 
            + inner(avg(inner(Rfacet, self.z_err)), self.both(test))*dS + 
            + inner(inner(Rfacet, self.z_err), test)*ds
        )
        
        # ------------------------------- Adjoint residual -------------------------------
        if s.use_adjoint_residual == True:
            (vF,) = self.F.arguments()  # test Argument used in self.F
            # r*(v) = J'(u)[v] - A'_u(u)[v, z]  since self.F = A(u;v) - L(v)
            rstar_form = derivative(self.J, self.u, vF) - derivative(replace(self.F, {vF: self.z}), self.u, vF)

            # dual: project r* → Rcell*, Rfacet*
            Lc_star = self.residual(rstar_form, bubbles*vc)   
            Rcell_star = Function(DG, name="Rcell_star")
            solve(ac == Lc_star, Rcell_star, solver_parameters=s.sp_cell)

            Lf_star = self.residual(rstar_form, Qtest) - inner(Rcell_star, Qtest)*dx
            Rhat_star = Function(Q)
            solve(af == Lf_star, Rhat_star, solver_parameters=s.sp_facet)
            Rfacet_star = Rhat_star/cones

            # indicators: 0.5 * (primal + dual)
            eta_dual = assemble(
                inner(inner(Rcell_star,   self.u_err), test)*dx
                + inner(avg(inner(Rfacet_star,    self.u_err)),self.both(test))*dS
                + inner(inner(Rfacet_star,  self.u_err), test)*ds
            )

            self.etaT = assemble(0.5*(eta_primal + eta_dual))
        else:
            self.etaT = eta_primal

        # Exact error indicators (experimental - ignore)
        if self.solverctx.exact_indicators == True:
            u_err_exact = self.u_exact - self.u
            eta_dual_exact = assemble(
                inner(inner(Rcell_star,   u_err_exact), test)*dx
                + inner(avg(inner(Rfacet_star,    u_err_exact)),self.both(test))*dS
                + inner(inner(Rfacet_star,  u_err_exact), test)*ds
            )
            udiff = assemble(eta_dual_exact - eta_dual)
            with udiff.dat.vec as uvec:
                unorm = uvec.norm()
            print("L2 error in (dual) refinement indicators: ", unorm)

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

        if self.u_exact is not None or self.goal_exact is not None:
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
        if self.u_exact is not None:
            def as_mixed(exprs):
                return as_vector([e[idx] for e in exprs for idx in np.ndindex(e.ufl_shape)])
            if type(self.u_exact) == list or type(self.u_exact) == tuple:
                u_exact_vec = as_mixed(self.u_exact)
                self.u_exact = refine(u_exact_vec, refine, coefficient_mapping=coef_map)
            else:
                self.u_exact = refine(self.u_exact, refine, coefficient_mapping=coef_map)
        self.problem = refine(self.problem, refine, coefficient_mapping=coef_map)
        self.J = refine(self.J, refine, coefficient_mapping=coef_map)
        self.F = self.problem.F
        self.u = self.problem.u
        self.bcs = self.problem.bcs
        self.V = self.u.function_space()
        self.problem = NonlinearVariationalProblem(self.F,self.u,self.bcs)
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
        if s.run_name is None:
            file_path = self.output_dir / "results.csv"
        else:
            file_path = self.output_dir / f"{s.run_name}/{s.run_name}_results.csv"
        if self.u_exact is None and self.goal_exact is None and self.solverctx.uniform_refinement == False:
            headers = ("iteration", "N", "Ndual", "Juh", "eta_h", "sum_eta_T")
            row = (
                it,
                self.N_vec[-1], self.Ndual_vec[-1], self.Juh, self.etah_vec[-1], self.etaTsum_vec[-1]
            )
        elif self.solverctx.uniform_refinement == True:
            headers = ("iteration", "N", "Ndual", "Juh", "eta", "eta_h")
            row = (
                it,
                self.N_vec[-1], self.Ndual_vec[-1], self.Juh, self.eta_vec[-1], self.etah_vec[-1]
            )
        else:
            headers = ("iteration", "N", "Ndual", "Juh", "eta", "eta_h", "sum_eta_T", "eff1", "eff2")
            row = (
                it,
                self.N_vec[-1], self.Ndual_vec[-1], self.Juh, self.eta_vec[-1], self.etah_vec[-1], self.etaTsum_vec[-1], self.eff1_vec[-1], self.eff2_vec[-1]
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
            VTKFile(self.output_dir / f"{s.run_name}/{s.run_name}_mesh_{it}.pvd").write(self.mesh)

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
            VTKFile(self.output_dir / f"{s.run_name}/{s.run_name}_solution_{it}.pvd").write(*self.u.subfunctions)
            print("Writing (dual) solution ...")
            VTKFile(self.output_dir / f"{s.run_name}/{s.run_name}_dual_solution_{it}.pvd").write(*self.z.subfunctions)

    def solve(self):
        s = self.solverctx

        for it in range(s.max_iterations):
            print(f"---------------------------- [MESH LEVEL {it}] ----------------------------")
            self.write_mesh(it)
            self.solve_primal()
            self.solve_dual()
            self.write_solution(it)
            self.compute_etah()
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

    # Utility functions
    def residual(self, F, test): # Residual helper function
        v = F.arguments()[0]
        return replace(F, {v: test})
    
    def both(self, u):
        return u("+") + u("-")

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


from ufl.duals import is_dual
from firedrake.dmhooks import get_transfer_manager

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

# Only required for my examples:
def getlabels(mesh, codim):
    ngmesh = mesh.netgen_mesh
    names = ngmesh.GetRegionNames(codim=codim)
    names_to_labels = {}
    for l in names:
        names_to_labels[l] = tuple(i+1 for i, name in enumerate(names) if name == l)
    return names_to_labels