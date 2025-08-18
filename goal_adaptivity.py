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
from adaptive_mg.adaptive import AdaptiveMeshHierarchy
from adaptive_mg.adaptive_transfer_manager import AdaptiveTransferManager

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

    def __init__(self, problem: NonlinearVariationalProblem, goal_functional, tolerance: float,  solver_parameters: dict, exact_solution = None, exact_goal = None):
        self.problem = problem
        self.solver_parameters = solver_parameters
        self.J = goal_functional
        self.tolerance = tolerance

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
        ndofs = self.V.dim()
        print("Primal dofs:" , ndofs)
        self.N_vec.append(ndofs)

        print("Solving primal ...")
        NonlinearVariationalSolver(self.problem).solve()

    def solve_dual(self):
        s = self.solverctx

        element = self.V.ufl_element()
        degree = element.degree()
        dual_degree = degree + 1
        dual_element = PMGPC.reconstruct_degree(element, dual_degree)
        Vdual = FunctionSpace(self.mesh, dual_element)
        vtest = TestFunction(Vdual) # Dual test function
        self.z = Function(Vdual) # Dual soluton

        ndofs_dual = Vdual.dim()
        print("Dual problem dofs:", ndofs_dual)
        self.Ndual_vec.append(ndofs_dual)    

        G = ( action(adjoint(derivative(self.F, self.u, TrialFunction(Vdual))), self.z) 
             - derivative(self.J, self.u, vtest) )

        # Dual ERROR - Implement today
        #eta_h_dual = abs(assemble(action(G, self.u_err)))
        #print("Etah dual = ", eta_h_dual)
        
        bcs_dual  = [bc.reconstruct(V=Vdual, indices=bc._indices, g=0) for bc in self.bcs]
        
        if s.dual_solve_method == "high_order":
            print("Solving dual ...")
            solve(G == 0, self.z, bcs_dual, solver_parameters=s.sp_chol) # Obtain z
            z_lo = Function(self.V, name="LowOrderDualSolution")
            z_lo.interpolate(self.z)
            self.z_err = self.z - z_lo
    
        elif s.dual_solve_method == "star":
            print("Solving dual ...")
            solve(G == 0, self.z, bcs_dual, solver_parameters=s.sp_star)
            self.z_err = self.z    
        
        else:
            print("ERROR: Unknown dual solve method.")

    def solve_u_err(self):
        element = self.V.ufl_element()
        degree = element.degree()
        high_order_element = PMGPC.reconstruct_degree(element, degree + 1)
        V_high_order = FunctionSpace(self.mesh, high_order_element)
        test = TestFunction(V_high_order) # Dual test function
        self.u_err = Function(V_high_order) # Dual soluton

        bcs_high_order  = [bc.reconstruct(V=V_high_order, indices=bc._indices) for bc in self.bcs]
        solve(self.F == 0, self.u_err, bcs_high_order)



    def compute_etah(self):
        # Compute error estimate F(z)
        self.eta_h = abs(assemble(self.residual(self.F, self.z)))
        # Add in average with adjoint residual G(u) for nonlinear problems

        # Append to state vectors for later
        self.etah_vec.append(self.eta_h)

        # Compute true error in J(uh)
        Juh = assemble(self.J)
        print(f"J(uh): {Juh}")

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
            print(f"J(u): {Ju}")
            print(f"eta = {self.eta}")

        if self.goal_exact is not None:
            Ju = self.goal_exact
            self.eta = abs(Juh - Ju)
            self.eta_vec.append(self.eta)

            # Print
            print(f"J(u): {Ju}")
            print(f"eta = {self.eta}")

        print(f"eta_h = {self.eta_h}")

    def automatic_error_indicators(self):
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
        print("Rcell dofs:" , ndofs)
        print("Computing Rcells ...")

        
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
        print("Rhat dofs:" , ndofs)
        print("Computing Rfacets ...")
        solve(af == Lf, Rhat, solver_parameters=s.sp_facet1)
        Rfacet = Rhat/cones

        # 8. Compute error indicators eta_T 
        DG0 = FunctionSpace(self.mesh, "DG", degree=0)
        test = TestFunction(DG0)

        print("Computing eta_T indicators ...")
        self.etaT = assemble(
            inner(inner(Rcell, self.z_err), test)*dx + 
            + inner(avg(inner(Rfacet, self.z_err)), both(test))*dS + 
            + inner(inner(Rfacet, self.z_err), test)*ds
        )
        return

    def manual_error_indicators(self): # Poisson ONLY!!!!!!!!!!
        s = self.solverctx
        n = FacetNormal(self.mesh)

        DG0 = FunctionSpace(self.mesh, "DG", degree=0)
        test = TestFunction(DG0)

        def both(u):
            return u("+") + u("-")
        
        print("Computing eta_T indicators ...")

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
        print(f"sum_T(eta_T): {self.etaT_total}")

        if self.u_exact is not None:
            # Compute efficiency indices
            self.eff1 = self.eta_h/self.eta
            self.eff2 = self.etaT_total/self.eta
            print(f"Efficiency index 1 = {self.eff1}")
            print(f"Efficiency index 2 = {self.eff2}")
            self.eff1_vec.append(self.eff1)
            self.eff2_vec.append(self.eff2)
        else:
            self.eff3 = self.etaT_total/self.eta_h
            print("Efficiency index, sum(eta_T)/eta_h = ", self.eff3)
            self.eff3_vec.append(self.eff3)

    def mark_and_refine(self):
        s = self.solverctx

        # 9. Mark cells for refinement (Dorfler marking)
        print("Marking cells for refinement ...")
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

        print("Refining mesh ...")
        new_mesh = self.mesh.refine_marked_elements(markers)
        print("Transferring problem to new mesh ...")
        amh = AdaptiveMeshHierarchy([self.mesh])
        atm = AdaptiveTransferManager()
        amh.add_mesh(new_mesh)
        coef_map = {}
        self.problem = coarsen(self.problem, coarsen, coefficient_mapping=coef_map)
        self.J = coarsen(self.J, coarsen, coefficient_mapping=coef_map)
        self.F = self.problem.F
        self.u = self.problem.u
        self.bcs = self.problem.bcs
        self.V = self.u.function_space()
        self.problem = NonlinearVariationalProblem(self.F,self.u,self.bcs)
        self.mesh = new_mesh
        if self.u_exact is not None:
            def as_mixed(exprs):
                return as_vector([e[idx] for e in exprs for idx in np.ndindex(e.ufl_shape)])
            if type(self.u_exact) == list or type(self.u_exact) == tuple:
                u_exact_vec = as_mixed(self.u_exact)
                self.u_exact = coarsen(u_exact_vec, coarsen, coefficient_mapping=coef_map)
            else:
                self.u_exact = coarsen(self.u_exact, coarsen, coefficient_mapping=coef_map)
            
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
            print("Writing primal ...")
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
                print("Writing data ...")
                self.append_data(it)

        if s.write_at_iteration == False:
            print("Writing data ...")
            self.write_data()

    # Utility functions
    def residual(self, F, test): # Residual helper function
        v = F.arguments()[0]
        return replace(F, {v: test})
    

class GAParameterContinuation(GoalAdaptiveNonlinearVariationalSolver): # Finish this today, easy
    def solve(self):
        s = self.solverctx

        for it in range(s.max_iterations):
            print(f"Solving on level {it}")

            print("Writing mesh ...")
            VTKFile(self.output_dir / f"mesh{it}.pvd").write(self.mesh)

            if it == 0:
                nu_init = s.parameter_init
                iterations = s.parameter_iterations
                nu_final = s.parameter_final
                nu = self.parameter
                nu_vals = np.logspace(np.log10(nu_init), np.log10(nu_final), iterations)
                
                for nu_val in nu_vals:
                    print(f"Primal iteration {nu_val}")

                    nu.assign(nu_val)
                    nu_in_form = extract_firedrake_constants(self.F)
                    print("Nu in F: ", nu_in_form)
                    self.solve_primal()
            else:
                self.solve_primal()
            
            print("Writing primal ...")
            VTKFile(self.output_dir / f"solution_{it}.pvd").write(*self.u.subfunctions)

            self.solve_dual()

            self.compute_etah()
            if self.eta_h < self.tolerance:
                print("Error estimate below tolerance, finished.")
                break

            if it == s.max_iterations -1:
                print(f"Maximum iteration ({s.max_iterations}) reached. Exiting.")
                break

            self.automatic_error_indicators()
            self.compute_efficiency()
            self.mark_and_refine()

            if s.write_at_iteration == True:
                print("Writing data ...")
                self.append_data(it)


def getlabels(mesh): # Doesn't seem to work in 2D ?
    ngmesh = mesh.netgen_mesh
    names = ngmesh.GetRegionNames(codim=1)
    print(names)
    names_to_labels = {}
    for l in names:
        names_to_labels[l] = tuple(i+1 for i, name in enumerate(names) if name == l)
        print(names_to_labels[l])
    return names_to_labels