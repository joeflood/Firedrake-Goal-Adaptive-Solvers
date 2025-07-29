from firedrake import *
from netgen.occ import *
import csv
from mesh_ctx import MeshCtx
from solver_ctx import SolverCtx
from problem_ctx import ProblemCtx
from typing import Callable, Any
from pathlib import Path
import numpy as np
from tsfc.ufl_utils import extract_firedrake_constants

class GoalAdaption:
    # Stores:
    # problemctx <- Changes
    # meshctx   <- Changes
    # solverctx <- Unchanged

    # State: (For each iteration)
    # u
    # z
    # z_err

    def __init__(self, meshctx: MeshCtx, problem_fn: Callable[[MeshCtx, SolverCtx], ProblemCtx], solverctx: SolverCtx):
        self.problemfn = problem_fn
        # Obtain initial mesh and solver ctx
        self.meshctx = meshctx
        self.solverctx = solverctx

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

        self.problemctx = self.problemfn(self.meshctx, self.solverctx)
        self.u = self.problemctx.u

    def build_problem(self):
        self.problemctx = self.problemfn(self.meshctx, self.solverctx)
        self.u = self.problemctx.u

    def solve_primal(self):
        p = self.problemctx

        ndofs = p.V.dim()
        print("Primal dofs:" , ndofs)
        self.N_vec.append(ndofs)

        print("Solving primal ...")
        solve(p.F == 0, self.u, bcs=p.bcs, solver_parameters={"snes_monitor": None})

    def solve_dual(self):
        m = self.meshctx
        s = self.solverctx
        p = self.problemctx

        element = p.V.ufl_element()
        degree = element.degree()
        dual_degree = degree + 1
        dual_element = PMGPC.reconstruct_degree(element, dual_degree)
        Vdual = FunctionSpace(m.mesh, dual_element)
        vtest = TestFunction(Vdual) # Dual test function
        self.z = Function(Vdual) # Dual soluton

        ndofs_dual = Vdual.dim()
        print("Dual problem dofs:", ndofs_dual)
        self.Ndual_vec.append(ndofs_dual)    

        G = ( action(adjoint(derivative(p.F, p.u, TrialFunction(Vdual))), self.z) 
             - derivative(p.J, p.u, vtest) )

        # Needs fixing      
        #eta_h_dual = abs(assemble(action(G, self.u)))
        #print("Etah dual = ", eta_h_dual)
        
        bcs_dual  = [bc.reconstruct(V=Vdual, indices=bc._indices, g=0) for bc in p.bcs]
        
        if s.dual_solve_method == "high_order":
            print("Solving dual ...")
            solve(G == 0, self.z, bcs_dual, solver_parameters=s.sp_chol) # Obtain z
            z_lo = Function(p.V, name="LowOrderDualSolution")
            z_lo.interpolate(self.z)
            self.z_err = self.z - z_lo
    
        elif s.dual_solve_method == "star":
            print("Solving dual ...")
            solve(G == 0, self.z, bcs_dual, solver_parameters=s.sp_star)
            self.z_err = self.z    
        
        else:
            print("ERROR: Unknown dual solve method.")

    def compute_etah(self):
        p = self.problemctx

        # Compute error estimate F(z)
        self.eta_h = abs(assemble(self.residual(p.F, self.z)))
        # Add in average with adjoint residual G(u) for nonlinear problems

        # Append to state vectors for later
        self.etah_vec.append(self.eta_h)

        # Compute true error in J(uh)
        Juh = assemble(p.J)
        print(f"J(uh): {Juh}")

        if p.u_exact is not None:
            def as_mixed(exprs):
                return as_vector([e[idx] for e in exprs for idx in np.ndindex(e.ufl_shape)])

            if type(p.u_exact) == list or type(p.u_exact) == tuple:
                Ju = assemble(replace(p.J, {p.u: as_mixed(p.u_exact)}))
            else:
                Ju = assemble(replace(p.J, {p.u: p.u_exact}))
            
            self.eta = abs(Juh - Ju)
            self.eta_vec.append(self.eta)

            # Print
            print(f"J(u): {Ju}")
            print(f"eta = {self.eta}")

        if p.goal_exact is not None:
            Ju = p.goal_exact
            self.eta = abs(Juh - Ju)
            self.eta_vec.append(self.eta)

            # Print
            print(f"J(u): {Ju}")
            print(f"eta = {self.eta}")

        print(f"eta_h = {self.eta_h}")

    def automatic_error_indicators(self):
        # 7. Compute cell and facet residuals R_T, R_\partialT
        m = self.meshctx
        s = self.solverctx
        p = self.problemctx

        variant = "integral" # Finite element type 

        # ---------------- Equation 4.6 to find cell residual Rcell -------------------------
        B = FunctionSpace(m.mesh, "B", m.dim+1, variant=variant) # Bubble function space
        bubbles = Function(B).assign(1) # Bubbles

        # Discontinuous function space of Rcell polynomials
        if p.V.value_shape == ():
            DG = FunctionSpace(m.mesh, "DG", s.residual_degree, variant=variant)
        else:
            DG = TensorFunctionSpace(m.mesh, "DG", s.residual_degree, variant=variant, shape=p.V.value_shape)
        uc = TrialFunction(DG)
        vc = TestFunction(DG)
        ac = inner(uc, bubbles*vc)*dx
        Lc = self.residual(p.F, bubbles*vc)

        Rcell = Function(DG, name="Rcell") # Rcell polynomial
        ndofs = DG.dim()
        print("Rcell dofs:" , ndofs)
        print("Computing Rcells ...")

        
        assemble(Lc)
        solve(ac == Lc, Rcell, solver_parameters=s.sp_cell2) # solve for Rcell polynonmial

        def both(u):
            return u("+") + u("-")

        # ---------------- Equation 4.8 to find facet residual Rfacet -------------------------
        FB = FunctionSpace(m.mesh, "FB", m.dim, variant=variant) # Cone function space
        cones = Function(FB).assign(1) # Cones

        el = BrokenElement(FiniteElement("FB", cell=m.cell, degree=s.residual_degree+m.dim, variant=variant))
        if p.V.value_shape == ():
            Q = FunctionSpace(m.mesh, el)
        else: 
            Q = TensorFunctionSpace(m.mesh, el, shape=p.V.value_shape)
        Qtest = TestFunction(Q)
        Qtrial = TrialFunction(Q)
        Lf = self.residual(p.F, Qtest) - inner(Rcell, Qtest)*dx
        af = both(inner(Qtrial/cones, Qtest))*dS + inner(Qtrial/cones, Qtest)*ds

        Rhat = Function(Q)
        ndofs = Q.dim()
        print("Rhat dofs:" , ndofs)
        print("Computing Rfacets ...")
        solve(af == Lf, Rhat, solver_parameters=s.sp_facet1)
        Rfacet = Rhat/cones

        # 8. Compute error indicators eta_T 
        DG0 = FunctionSpace(m.mesh, "DG", degree=0)
        test = TestFunction(DG0)

        print("Computing eta_T indicators ...")
        self.etaT = assemble(
            inner(inner(Rcell, self.z_err), test)*dx + 
            + inner(avg(inner(Rfacet, self.z_err)), both(test))*dS + 
            + inner(inner(Rfacet, self.z_err), test)*ds
        )
        return

    # def manual_error_indicators(self): # Poisson ONLY!!!!!!!!!!
    #     m = self.meshctx
    #     s = self.solverctx
    #     p = self.problemctx

    #     DG0 = FunctionSpace(m.mesh, "DG", degree=0)
    #     test = TestFunction(DG0)

    #     def both(u):
    #         return u("+") + u("-")
        
    #     print("Computing eta_T indicators ...")
    #     self.etaT = assemble(
    #         inner(p.f + div(grad(self.u)), self.z_err * test) * dx +
    #         inner(0.5*jump(-grad(self.u), m.n), self.z_err * both(test)) * dS +
    #         inner(dot(-grad(self.u), m.n) + p.g, self.z_err * test) * p.ds_neumann
    #     )

    def compute_efficiency(self):
        p = self.problemctx

        with self.etaT.dat.vec as evec:
            evec.abs()    
            self.etaT_array = evec.getArray()

        self.etaT_total = abs(np.sum(self.etaT_array))
        self.etaTsum_vec.append(self.etaT_total)
        print(f"sum_T(eta_T): {self.etaT_total}")

        if p.u_exact is not None:
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
        m = self.meshctx
        s = self.solverctx
        p = self.problemctx        
        # 9. Mark cells for refinement (Dorfler marking)
        print("Marking cells for refinement ...")
        sorted_indices = np.argsort(-self.etaT_array)
        sorted_etaT = self.etaT_array[sorted_indices]
        cumulative_sum = np.cumsum(sorted_etaT)
        threshold = s.dorfler_alpha * self.etaT_total
        M = np.searchsorted(cumulative_sum, threshold) + 1
        marked_cells = sorted_indices[:M]

        markers_space = FunctionSpace(m.mesh, "DG", 0)
        markers = Function(markers_space)
        with markers.dat.vec as mv:
            marr = mv.getArray()
            marr[:] = 0
            marr[marked_cells] = 1

        print("Refining mesh ...")
        m.update_mesh(m.mesh.refine_marked_elements(markers))

    def write_data(self):
        # Write to file
        rows = list(zip(self.N_vec, self.eta_vec, self.etah_vec, self.etaTsum_vec, self.eff1_vec, self.eff2_vec))
        headers = ("N", "eta", "eta_h", "sum_eta_T", "eff1", "eff2")
        with open(self.output_dir / "results.csv", "w", newline="") as file:
            w = csv.writer(file)
            w.writerow(headers)
            w.writerows(rows)   

    def solve(self):
        m = self.meshctx
        s = self.solverctx

        for it in range(s.max_iterations):
            print(f"Solving on level {it}")

            print("Writing mesh ...")
            VTKFile(self.output_dir / f"mesh{it}.pvd").write(m.mesh)

            if it != 0:
                print("Defining problem on new mesh ...")
                self.build_problem() # Redefine problem on new mesh

            self.solve_primal()
            print("Writing primal ...")
            VTKFile(self.output_dir / f"solution_{it}.pvd").write(*self.u.subfunctions)

            self.solve_dual()

            self.compute_etah()
            if self.eta_h < s.tolerance:
                print("Error estimate below tolerance, finished.")
                break

            if it == s.max_iterations -1:
                print(f"Maximum iteration ({s.max_iterations}) reached. Exiting.")
                break

            self.automatic_error_indicators()
            self.compute_efficiency()
            self.mark_and_refine()

        print("Writing data ...")
        self.write_data()

    # Utility functions
    def residual(self, F, test): # Residual helper function
        v = F.arguments()[0]
        return replace(F, {v: test})
    

class GoalAdaptionStabilized(GoalAdaption):
    def solve(self):
        m = self.meshctx
        s = self.solverctx
        p = self.problemctx

        for it in range(s.max_iterations):
            print(f"Solving on level {it}")

            print("Writing mesh ...")
            VTKFile(self.output_dir / f"mesh{it}.pvd").write(m.mesh)

            print("Defining problem on new mesh ...")
            self.build_problem() # Redefine problem on new mesh

            if it == 0:
                nu_init = s.parameter_init
                iterations = s.parameter_iterations
                nu_final = s.parameter_final
                nu = p.parameter
                nu_vals = np.logspace(np.log10(nu_init), np.log10(nu_final), iterations)
                
                for nu_val in nu_vals:
                    print(f"Primal iteration {nu_val}")

                    nu.assign(nu_val)
                    nu_in_form = extract_firedrake_constants(p.F)
                    print("Nu in F: ", nu_in_form)
                    self.solve_primal()
            else:
                self.solve_primal()
            
            print("Writing primal ...")
            VTKFile(self.output_dir / f"solution_{it}.pvd").write(*self.u.subfunctions)

            self.solve_dual()

            self.compute_etah()
            if self.eta_h < s.tolerance:
                print("Error estimate below tolerance, finished.")
                break

            if it == s.max_iterations -1:
                print(f"Maximum iteration ({s.max_iterations}) reached. Exiting.")
                break

            self.automatic_error_indicators()
            self.compute_efficiency()
            self.mark_and_refine()

        print("Writing data ...")
        self.write_data()