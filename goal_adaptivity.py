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
    Solves a goal adaption problem.
    Stores:
    solverctx: Keep? Look at what Firedrake solve functions do

    State: (For each iteration)
    u
    z
    z_err
    u_err ? Soon - for dual sol.            
    '''

    def __init__(self, problem: NonlinearVariationalProblem, goal_functional, tolerance: float,  solver_parameters: dict,*, primal_solver_parameters = None, dual_solver_parameters = None, exact_solution = None, exact_goal = None):
        self.problem = problem
        self.solver_parameters = solver_parameters
        self.J = goal_functional
        self.tolerance = tolerance
        self.sp_primal = primal_solver_parameters
        self.sp_dual = dual_solver_parameters

        self.V = problem.u.function_space()
        self.u = problem.u
        self.bcs = problem.bcs
        self.F = problem.F
        self.u_exact = exact_solution
        self.goal_exact = exact_goal
        # We also need other things
        self.element = self.V.ufl_element()
        self.test = TestFunction(self.V)
        self.mesh = self.V.mesh()
        self.solverctx = SolverCtx(solver_parameters) # To store solver data (Maybe remove?)

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
        print(f"Solving primal (degree: {self.solverctx.degree}, dofs: {ndofs}) ...")
        if self.sp_primal is None:
            print("Method: Default nonlinear solve")
            NonlinearVariationalSolver(self.problem).solve()
        else:
            print("Method: User defined")
            NonlinearVariationalSolver(self.problem, solver_parameters=self.sp_primal).solve()
        
        if self.solverctx.residual == "both":
            element = self.V.ufl_element()
            degree = element.degree()
            high_element = PMGPC.reconstruct_degree(element, degree + 1)
            Vhigh = FunctionSpace(self.mesh, high_element)
            test = TestFunction(Vhigh) # Dual test function
            self.u_high = Function(Vhigh) # Dual soluton
            (v_old,) = self.F.arguments()
            v_high = TestFunction(Vhigh)
            F_high = ufl.replace(self.F, {v_old: v_high, self.u: self.u_high})
            bcs_high = reconstruct_bcs(self.bcs, Vhigh)
            self.problem_high = NonlinearVariationalProblem(F_high, self.u_high, bcs_high)
            
            print(f"Solving primal in higher space for error estimate (degree: {degree + 1}, dofs: {Vhigh.dim()}) ...")
            if self.sp_primal is None:
                print("Method: Default nonlinear solve")
                NonlinearVariationalSolver(self.problem_high).solve()
            else:
                print("Method: User defined")
                NonlinearVariationalSolver(self.problem_high, solver_parameters=self.sp_primal).solve()
            
            #self.u.project(self.u_high) gives BAD results
            self.u_err = self.u_high - self.u

    def solve_dual(self):
        s = self.solverctx

        element = self.V.ufl_element()
        degree = element.degree()
        dual_element = PMGPC.reconstruct_degree(element, degree +1)
        Vdual = FunctionSpace(self.mesh, dual_element)
        vtest = TestFunction(Vdual) # Dual test function
        self.z = Function(Vdual) # Dual soluton

        ndofs_dual = Vdual.dim()
        self.Ndual_vec.append(ndofs_dual)    

        self.G = ( action(adjoint(derivative(self.F, self.u, TrialFunction(Vdual))), self.z) 
             - derivative(self.J, self.u, vtest) )
        
        bcs_dual  = [bc.reconstruct(V=Vdual, indices=bc._indices, g=0) for bc in self.bcs]
        
        print(f"Solving dual (degree: {degree + 1}, dofs: {ndofs_dual}) ...")
        if s.dual_solve_method == "star":
            print("Method: Vertex star relaxation")
            solve(self.G == 0, self.z, bcs_dual, solver_parameters=s.sp_star)

        elif s.dual_solve_method == "vanka":
            print("Method: Vanka relaxation")
            solve(self.G == 0, self.z, bcs_dual, solver_parameters=s.sp_vanka)

        elif s.dual_solve_method == "cholesky":
            print("Method: Cholesky")
            solve(self.G == 0, self.z, bcs_dual, solver_parameters=s.sp_chol)
                  
        elif self.sp_dual is not None:
            print("Method: User defined")
            solve(self.G == 0, self.z, bcs_dual, solver_parameters=self.sp_dual)
            
        else:
            print("Method: Default nonlinear solve")
            solve(self.G == 0, self.z, bcs_dual)
            
        self.z_lo = Function(self.V, name="LowOrderDualSolution")
        self.z_lo.interpolate(self.z)
        self.z_err = self.z - self.z_lo

    def compute_etah(self):
        # Compute error estimate F(z)
        if self.solverctx.residual == "both":
            print("Predicting error as average of primal and dual:")
            primal_err = abs(assemble(self.residual(self.F, self.z_err)))

            G = ( action(adjoint(derivative(self.F, self.u, TrialFunction(self.V))), self.z_lo) 
             - derivative(self.J, self.u, TestFunction(self.V)) )
            dual_err = abs(assemble(self.residual(G, self.u_err)))
            print(f"{'Primal error, |ρ(u_h;z-z_h)|:':50s}{primal_err:20.12f}")
            print(f"{'Dual error, |ρ*(z_h;u-u_h)|:':50s}{dual_err:20.12f}")
            self.eta_h = abs(0.5* primal_err + 0.5*dual_err)
            print(f"Difference between primal and dual errors: {abs(primal_err-dual_err)}")
        else:
            self.eta_h = abs(assemble(self.residual(self.F, self.z_err)))
        # Add in average with adjoint residual G(u) for nonlinear problems

        # Append to state vectors for later
        self.etah_vec.append(self.eta_h)

        # Compute true error in J(uh)
        Juh = assemble(self.J)
        print(f"{'Computed goal, J(uh):':50s}{Juh:20.12f}")

        if self.u_exact is not None:
            def as_mixed(exprs):
                return as_vector([e[idx] for e in exprs for idx in np.ndindex(e.ufl_shape)])

            if type(self.u_exact) == list or type(self.u_exact) == tuple:
                Ju = assemble(replace(self.J, {self.u: as_mixed(self.u_exact)}))
            else:
                Ju = assemble(replace(self.J, {self.u: self.u_exact}))
            
            self.eta = abs(Juh - Ju)
            self.eta_vec.append(self.eta)

            # Print
            print(f"{'Exact goal, J(u):':50s}{Ju:20.12f}")
            print(f"{'True error, η = |J(u) - J(u_h)|:':50s}{self.eta:20.12f}")

        if self.goal_exact is not None:
            Ju = self.goal_exact
            self.eta = abs(Juh - Ju)
            self.eta_vec.append(self.eta)

            # Print
            print(f"{'Exact goal J(u):':50s}{Ju:20.12f}")
            print(f"{'True error η = |J(u) - J(u_h)|:':50s}{self.eta:20.12f}")
        if self.solverctx.residual == "both":
            print(f"{'Predicted error, η_h = |0.5ρ(u_h;z-z_h) + 0.5ρ*(z_h;u-u_h)|:':50s}{self.eta_h:20.12f}")
        else:
            print(f"{'Predicted error, η_h = |ρ(u_h;z-z_h)|:':50s}{self.eta_h:20.12f}")


    def automatic_error_indicators(self):
        print("Computing local refinement indicators, η_K...")
        # 7. Compute cell and facet residuals R_T, R_\partialT
        s = self.solverctx
        dim = self.mesh.topological_dimension()
        cell = self.mesh.ufl_cell()

        variant = "integral" # Finite element type 

        # ---------------- Equation 4.6 to find cell residual Rcell -------------------------
        B = FunctionSpace(self.mesh, "B", dim+1, variant=variant) # Bubble function space
        bubbles = Function(B).assign(1) # Bubbles

        # Discontinuous function space of Rcell polynomials
        if self.V.value_shape == ():
            DG = FunctionSpace(self.mesh, "DG", s.residual_degree, variant=variant)
        else:
            DG = TensorFunctionSpace(self.mesh, "DG", s.residual_degree, variant=variant, shape=self.V.value_shape)

        uc = TrialFunction(DG)
        vc = TestFunction(DG)
        ac = inner(uc, bubbles*vc)*dx
        Lc = self.residual(self.F, bubbles*vc)

        Rcell = Function(DG, name="Rcell") # Rcell polynomial
        ndofs = DG.dim()
        #print("Rcell dofs:" , ndofs)
        #print("Computing Rcells ...")

        
        assemble(Lc)
        solve(ac == Lc, Rcell, solver_parameters=s.sp_cell2) # solve for Rcell polynonmial

        def both(u):
            return u("+") + u("-")

        # ---------------- Equation 4.8 to find facet residual Rfacet -------------------------
        FB = FunctionSpace(self.mesh, "FB", dim, variant=variant) # Cone function space
        cones = Function(FB).assign(1) # Cones

        el = BrokenElement(FiniteElement("FB", cell=cell, degree=s.residual_degree+dim, variant=variant))
        if self.V.value_shape == ():
            Q = FunctionSpace(self.mesh, el)
        else: 
            Q = TensorFunctionSpace(self.mesh, el, shape=self.V.value_shape)
        Qtest = TestFunction(Q)
        Qtrial = TrialFunction(Q)
        Lf = self.residual(self.F, Qtest) - inner(Rcell, Qtest)*dx
        af = both(inner(Qtrial/cones, Qtest))*dS + inner(Qtrial/cones, Qtest)*ds

        Rhat = Function(Q)
        ndofs = Q.dim()
        #print("Rhat dofs:" , ndofs)
        #print("Computing Rfacets ...")
        solve(af == Lf, Rhat, solver_parameters=s.sp_facet1)
        Rfacet = Rhat/cones
        
        # 8. Compute error indicators eta_T 
        DG0 = FunctionSpace(self.mesh, "DG", degree=0)
        test = TestFunction(DG0)

        eta_primal = assemble(
            inner(inner(Rcell, self.z_err), test)*dx + 
            + inner(avg(inner(Rfacet, self.z_err)), both(test))*dS + 
            + inner(inner(Rfacet, self.z_err), test)*ds
        )
        
        if s.residual == "both":
            # ---------- ★ dual residual form r*(·) ----------
            (vF,) = self.F.arguments()  # test Argument used in self.F
            # r*(v) = J'(u)[v] - A'_u(u)[v, z]  since self.F = A(u;v) - L(v)
            rstar_form = derivative(self.J, self.u, vF) - derivative(replace(self.F, {vF: self.z}), self.u, vF)

            # ---------- ★ dual: project r* → Rcell*, Rfacet* ----------
            Lc_star = self.residual(rstar_form, bubbles*vc)            # same matrix ac
            Rcell_star = Function(DG, name="Rcell_star")
            solve(ac == Lc_star, Rcell_star, solver_parameters=s.sp_cell2)

            Lf_star = self.residual(rstar_form, Qtest) - inner(Rcell_star, Qtest)*dx
            Rhat_star = Function(Q)
            solve(af == Lf_star, Rhat_star, solver_parameters=s.sp_facet1)
            Rfacet_star = Rhat_star/cones

            # ---------- indicators: 0.5 * (primal + dual) ----------
            eta_dual = assemble(
                inner(inner(Rcell_star,   self.u_err), test)*dx
                + inner(avg(inner(Rfacet_star,    self.u_err)),both(test))*dS
                + inner(inner(Rfacet_star,  self.u_err), test)*ds
            )

            self.etaT = assemble(0.5*(eta_primal + eta_dual))
    
        else:
            self.etaT = eta_primal

        if self.solverctx.exact_indicators == True:
            u_err_exact = self.u_exact - self.u
            eta_dual_exact = assemble(
                inner(inner(Rcell_star,   u_err_exact), test)*dx
                + inner(avg(inner(Rfacet_star,    u_err_exact)),both(test))*dS
                + inner(inner(Rfacet_star,  u_err_exact), test)*ds
            )
            udiff = assemble(eta_dual_exact - eta_dual)
            with udiff.dat.vec as uvec:
                unorm = uvec.norm()
            print("L2 error in (dual) refinement indicators: ", unorm)

    def manual_error_indicators(self): # Poisson ONLY!!!!!!!!!!
        print("Computing local refinement indicators (η_K)...")
        s = self.solverctx
        n = FacetNormal(self.mesh)

        DG0 = FunctionSpace(self.mesh, "DG", degree=0)
        test = TestFunction(DG0)

        def both(u):
            return u("+") + u("-")

        self.etaT = assemble(
            inner(self.f + div(grad(self.u)), self.z_err * test) * dx +
            inner(0.5*jump(-grad(self.u), n), self.z_err * both(test)) * dS +
            inner(dot(-grad(self.u), n), self.z_err * test) * ds
        )

    def compute_efficiency(self):
        with self.etaT.dat.vec as evec:
            evec.abs()    
            self.etaT_array = evec.getArray()

        self.etaT_total = abs(np.sum(self.etaT_array))
        self.etaTsum_vec.append(self.etaT_total)
        print(f"{'sum of refinement indicators, sum η_K:':50s}{self.etaT_total:20.12f}")

        if self.u_exact is not None or self.goal_exact is not None:
            # Compute efficiency indices
            self.eff1 = self.eta_h/self.eta
            self.eff2 = self.etaT_total/self.eta
            print(f"{'Effectivity index 1, η_h/η:':50s}{self.eff1:14.6}")
            print(f"{'Effectivity index 2, sum η_K/η:':50s}{self.eff2:14.6f}")
            self.eff1_vec.append(self.eff1)
            self.eff2_vec.append(self.eff2)
        else:
            self.eff3 = self.etaT_total/self.eta_h
            print(f"{'Effectivity index, sum η_K/η_h):':50s}{self.eff3:14.6f}")
            self.eff3_vec.append(self.eff3)

    def mark_and_refine(self):
        s = self.solverctx

        # 9. Mark cells for refinement (Dorfler marking)
        print("Refining mesh ...")
        sorted_indices = np.argsort(-self.etaT_array)
        sorted_etaT = self.etaT_array[sorted_indices]
        cumulative_sum = np.cumsum(sorted_etaT)
        threshold = s.dorfler_alpha * self.etaT_total
        M = np.searchsorted(cumulative_sum, threshold) + 1
        marked_cells = sorted_indices[:M]

        markers_space = FunctionSpace(self.mesh, "DG", 0)
        markers = Function(markers_space)
        with markers.dat.vec as mv:
            marr = mv.getArray()
            marr[:] = 0
            marr[marked_cells] = 1
        new_mesh = self.mesh.refine_marked_elements(markers)

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
        # Write to file
        rows = list(zip(self.N_vec, self.Ndual_vec, self.eta_vec, self.etah_vec, self.etaTsum_vec, self.eff1_vec, self.eff2_vec))
        headers = ("N", "Ndual", "eta", "eta_h", "sum_eta_T", "eff1", "eff2")
        with open(self.output_dir / "results.csv", "w", newline="") as file:
            w = csv.writer(file)
            w.writerow(headers)
            w.writerows(rows)
            jump

    def append_data(self, it):
        file_path = self.output_dir / "results.csv"
        if self.u_exact is not None:
            headers = ("iteration", "N", "Ndual", "eta", "eta_h", "sum_eta_T", "eff1", "eff2")
            row = (
                it,
                self.N_vec[-1], self.Ndual_vec[-1], self.eta_vec[-1], self.etah_vec[-1], self.etaTsum_vec[-1], self.eff1_vec[-1], self.eff2_vec[-1]
            )
        else:
            headers = ("iteration", "N", "Ndual", "eta_h", "sum_eta_T")
            row = (
                it,
                self.N_vec[-1], self.Ndual_vec[-1], self.etah_vec[-1], self.etaTsum_vec[-1]
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
        

    def solve(self):
        s = self.solverctx

        for it in range(s.max_iterations):
            print(f"Solving on level {it}")

            print("Writing mesh ...")
            VTKFile(self.output_dir / f"mesh{it}.pvd").write(self.mesh)

            self.solve_primal()
            print("Writing primal solution ...")
            VTKFile(self.output_dir / f"solution_{it}.pvd").write(*self.u.subfunctions)

            self.solve_dual()

            self.compute_etah()
            if self.eta_h < self.tolerance:
                print("Error estimate below tolerance, finished.")
                break

            if it == s.max_iterations -1:
                print(f"Maximum iteration ({s.max_iterations}) reached. Exiting.")
                break
            

            if s.residual_solve_method == "automatic":
                self.automatic_error_indicators()
            elif s.residual_solve_method == "manual":
                self.manual_error_indicators()
            else:
                print("Unknown residual solve method. Exiting.")
                break
            
            self.compute_efficiency()
            self.mark_and_refine()

            if s.write_at_iteration == True:
                print("Appending data ...")
                self.append_data(it)

        if s.write_at_iteration == False:
            print("Writing data ...")
            self.write_data()

    # Utility functions
    def residual(self, F, test): # Residual helper function
        v = F.arguments()[0]
        return replace(F, {v: test})

def getlabels(mesh): # Doesn't seem to work in 2D ?
    ngmesh = mesh.netgen_mesh
    names = ngmesh.GetRegionNames(codim=1)
    names_to_labels = {}
    for l in names:
        names_to_labels[l] = tuple(i+1 for i, name in enumerate(names) if name == l)
    return names_to_labels

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