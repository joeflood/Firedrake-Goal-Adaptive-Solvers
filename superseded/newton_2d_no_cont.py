# This script solves the 2D SOSM problem using Newton's method
# For simplicity only consider the two species case
# The parameter g_ms_const is given as input from the terminal

import sys
import shutil
from firedrake import *
from firedrake.output import VTKFile
from firedrake.petsc import PETSc
import firedrake.utils as firedrake_utils
import petsc4py
import numpy as np
import numpy.linalg as npla
from tabulate import tabulate

# -------------- Some key parameters relating to the problem setup -------------

d = 2                       # The spatial dimension
assert d == 2

k = 4                       # The polynomial degree (for the velocity spaces)
deg_max = 8 * k             # Maximum quadrature degree

mesh_type = "quad"          # What mesh type to use
assert mesh_type in ["triangle", "quad"]

# -------------- Looping over meshes to get convergence rates-------------------

n_loops = 5

num_err_fields = 16
errs = np.empty((n_loops, num_err_fields, 2))

for i in range(0, n_loops):

    # ----------------------------- The computational mesh ---------------------

    N_mesh = 4 * (2 ** i)

    if mesh_type == "triangle":
        mesh = UnitSquareMesh(N_mesh, N_mesh, diagonal="crossed")
    elif mesh_type == "quad":
        mesh = UnitSquareMesh(N_mesh, N_mesh, quadrilateral=True)

    vol = assemble(1.0*dx(mesh))

    # Utility function for L^2 norm and distance
    def l2_norm(f, bdry=False):
        if bdry:
            return np.sqrt(assemble(inner(f, f) * ds(mesh, degree=deg_max)))
        else:
            return np.sqrt(assemble(inner(f, f) * dx(mesh, degree=deg_max)))

    def l2_dist(f, g, bdry=False):
        return l2_norm(f - g, bdry)

    # Utility class for fixing a (possibly DG) field at a node
    class FixAtPointBC(DirichletBC):
        def __init__(self, V, g, sub_domain, bc_point):
            super().__init__(V, g, sub_domain)
            self.g = g
            self.bc_point = bc_point
            self.min_index = None

        @firedrake_utils.cached_property
        def nodes(self):
            V = self.function_space()
            x_sc = SpatialCoordinate(V.mesh())
            point_dist = Function(V)
            point_dist.interpolate(sqrt(dot(x_sc - self.bc_point, x_sc - self.bc_point)))

            with point_dist.dat.vec as v:
                min_index, min_value = v.min()
                assert min_value < 1e-8

                v.zeroEntries()
                v.setValue(min_index, 1.0)

                self.min_index = min_index

            nodes, = np.nonzero(point_dist.dat.data_ro_with_halos)
            assert len(nodes) <= 1
            nodes = np.asarray([nodes[0]] if (len(nodes) > 0) else [], dtype=int)

            nodes_lgmap = V.dof_dset.lgmap.applyInverse([min_index])
            nodes_lgmap = nodes_lgmap[nodes_lgmap >= 0]
            assert len(nodes_lgmap) <= 1

            assert set(nodes) == set(nodes_lgmap)
            return nodes

        def eval_at_point(self, V_func):
            if self.min_index is None:
                self.nodes      # To ensure that self.min_index is correctly initialized
            V_func_value = V_func.vector().gather(global_indices=[self.min_index])[0]
            return V_func_value

        def assert_is_enforced(self, V_func):
            bc_value = self.g(self.bc_point)
            V_func_value = self.eval_at_point(V_func)

            assert abs(bc_value - V_func_value) < 1e-8

        def dof_index_in_mixed_space(self, M, l):
            x_sc = SpatialCoordinate(M.mesh())
            dist_func = Function(M.sub(l))
            dist_func.interpolate(Constant(-1.0) + sqrt(dot(x_sc - self.bc_point, x_sc - self.bc_point)))

            func_mixed = Function(M)
            func_mixed.subfunctions[l].assign(dist_func)

            with func_mixed.dat.vec as v:
                v.shift(1.0)
                min_index, min_value = v.min()
                assert min_value < 1e-8

            return min_index

    # ----------------------------- The problem parameters ---------------------

    # Physical parameters

    # IMPORTANT: Molar masses must be one for the manufactured solution to work
    M_1 = Constant(1)           # Molar mass of species 1
    M_2 = Constant(1)           # Molar mass of species 2

    # IMPORTANT: RT must be one for the manufactured solution to work
    RT = Constant(1)            # Ideal gas constant times the ambient temperature

    # IMPORTANT: Need to define D_12 in this special way for the manufactured solution to work
    D_1 = Constant(0.5)         # Parameter for defining the Stefan-Maxwell diffusivity
    D_2 = Constant(2.0)         # Parameter for defining the Stefan-Maxwell diffusivity
    D_12 = D_1 * D_2            # Stefan-Maxwell diffusivity (= D_21)

    zeta = Constant(1e-1)                       # Bulk viscosity for Stokes
    eta = Constant(1e-1)                        # Shear viscosity for Stokes
    lame = (zeta - ((2.0 * eta) / d))           # Lame parameter

    # Other parameters
    gamma = Constant(1e1)                       # The augmentation parameter

    newton_atol = 1e-12                         # Newton absolute tolerance
    newton_max_it = 10                          # Newton maximum iterations

    # --------------------- The thermodynamic constitutive law -----------------

    # The ideal gas law (gives concentration as a function of chemical potential)
    def ideal_gas(mu):
        return exp(mu)

    # ----------------------------- The manufactured solution ------------------
    x = SpatialCoordinate(mesh)

    nml = FacetNormal(mesh)
    Id = Identity(d)

    g_ms_const = Constant(float(sys.argv[1]))       # g_mst_const is supplied in the terminal
    g_ms = g_ms_const * x[0] * x[1] * (1.0 - x[0]) * (1.0 - x[1])

    mu_1_ms = g_ms / D_1                # Chemical potential of species 1
    mu_2_ms = g_ms / D_2                # Chemical potential of species 2

    c_1_ms = exp(mu_1_ms)               # Concentration of species 1
    c_2_ms = exp(mu_2_ms)               # Concentration of species 2

    c_T_ms = c_1_ms + c_2_ms                    # Total concentration
    rho_ms = (M_1 * c_1_ms) + (M_2 * c_2_ms)    # Density

    omega_1_ms = (M_1 * c_1_ms) / rho_ms        # Mass fraction of species 1
    omega_2_ms = (M_2 * c_2_ms) / rho_ms        # Mass fraction of species 2

    v_1_ms = D_1 * grad(g_ms)                               # Velocity of species 1
    v_2_ms = D_2 * grad(g_ms)                               # Velocity of species 2
    v_ms = (omega_1_ms * v_1_ms) + (omega_2_ms * v_2_ms)    # Mass average velocity

    mm_1_ms = M_1 * c_1_ms * v_1_ms                         # Momentum of species 1
    mm_2_ms = M_2 * c_2_ms * v_2_ms                         # Momentum of species 2

    epsilon_v_ms = sym(grad(v_ms))                                  # Strain rate
    tau_ms = (2.0 * eta * epsilon_v_ms) + \
             lame * tr(epsilon_v_ms) * Id                           # Viscous stress
    p_ms = c_T_ms                                                   # Pressure

    f_ms = (grad(p_ms) - div(tau_ms))                               # Stokes forcing term (times the density)

    # ----------------------------- The problem data ---------------------------
    f = f_ms / rho_ms                           # Stokes forcing term

    r_1 = div(c_1_ms * v_1_ms)                  # Volumetric generation/depletion rate of species 1
    r_2 = div(c_2_ms * v_2_ms)                  # Volumetric generation/depletion rate of species 2

    g_v = v_ms                                  # Boundary condition for the bulk velocity

    g_1 = mm_1_ms                               # (Normal) boundary condition for species 1 momentum
    g_2 = mm_2_ms                               # (Normal) boundary condition for species 2 momentum

    # The boundary point that is used to remove the scalar field nullspaces
    bc_point_ref = Constant([0.0, 0.0])

    # The reference pressure, which is imposed at the boundary point
    p_ref = Constant(p_ms(bc_point_ref))

    # The exact pressure mean, which is only used after Newton's method has converged
    exact_p_mean = Constant(assemble(p_ms * dx(mesh, degree=deg_max)))

    # The total concentrations, which are enforced using the Woodbury identity
    c_1_total = Constant(assemble(c_1_ms * dx(mesh, degree=deg_max)))
    c_2_total = Constant(assemble(c_2_ms * dx(mesh, degree=deg_max)))

    # Print the norms of the problem data
    PETSc.Sys.Print("L^2 norms of the forcing terms: f %.2e, r_1 %.2e, r_2 %.2e" % 
                    (l2_norm(f), l2_norm(r_1), l2_norm(r_2)))
    
    PETSc.Sys.Print("L^2 norms of the boundary data: g_v %.2e, g_1 %.2e, g_2 %.2e" % 
                    (l2_norm(g_v, bdry=True), l2_norm(dot(g_1, nml), bdry=True), l2_norm(dot(g_2, nml), bdry=True)))
    
    PETSc.Sys.Print("Absolute values of the nullspace constraint data: p_ref %.2e, p_mean %.2e, c_1_total %.2e, c_2_total %.2e" % 
                    (abs(p_ref), abs(exact_p_mean), c_1_total, c_2_total))

    # ----------------- Solver parameters used for projections -----------------
    project_solver_parameters = {"ksp_type" : "gmres",
                                 "ksp_max_it" : 3,
                                 "ksp_convergence_test" : "skip",
                                 "pc_type" : "lu",
                                 "pc_factor_mat_solver_type" : "mumps",
                                 "mat_mumps_icntl_14": 105}

    project_fc_parameters = {"quadrature_degree" : deg_max}

    # Utility projection function with custom settings
    def project_util(expr, space, bcs=None):
        return project(expr, space, bcs=bcs, solver_parameters=project_solver_parameters, form_compiler_parameters=project_fc_parameters)

    # ----------------------------- The discrete spaces ------------------------

    if mesh_type == "triangle":
        var = "equispaced"
        W_h_1 = FunctionSpace(mesh, "RT", k)                                                    # Momentum space for species 1
        W_h_2 = FunctionSpace(mesh, "RT", k)                                                    # Momentum space for species 2
        V_h   = VectorFunctionSpace(mesh, "CG", k)                                              # Bulk velocity space
        X_h_1 = FunctionSpace(mesh, FiniteElement("DG", triangle, k - 1, variant=var))          # Chemical potential space for species 1
        X_h_2 = FunctionSpace(mesh, FiniteElement("DG", triangle, k - 1, variant=var))          # Chemical potential space for species 2
        P_h   = FunctionSpace(mesh, "CG", k - 1)                                                # Pressure space
        C_h_1 = FunctionSpace(mesh, "DG", k - 1)                                                # Concentration space for species 1
        C_h_2 = FunctionSpace(mesh, "DG", k - 1)                                                # Concentration space for species 2
        R_h = FunctionSpace(mesh, "CG", k - 1)                                                  # Density reciprocal space

    elif mesh_type == "quad":
        var = "equispaced"
        W_h_1 = FunctionSpace(mesh, FiniteElement("RTCF", quadrilateral, k, variant=var))       # Momentum space for species 1
        W_h_2 = FunctionSpace(mesh, FiniteElement("RTCF", quadrilateral, k, variant=var))       # Momentum space for species 2
        V_h   = VectorFunctionSpace(mesh, "CG", k)                                              # Bulk velocity space
        X_h_1 = FunctionSpace(mesh, FiniteElement("DQ", quadrilateral, k - 1, variant=var))     # Chemical potential space for species 1
        X_h_2 = FunctionSpace(mesh, FiniteElement("DQ", quadrilateral, k - 1, variant=var))     # Chemical potential space for species 2
        P_h   = FunctionSpace(mesh, "CG", k - 1)                                                # Pressure space
        C_h_1 = FunctionSpace(mesh, "DQ", k - 1)                                                # Concentration space for species 1
        C_h_2 = FunctionSpace(mesh, "DQ", k - 1)                                                # Concentration space for species 2
        R_h = FunctionSpace(mesh, "CG", k - 1)                                                  # Density reciprocal space

    Z_h = W_h_1 * W_h_2 * V_h * X_h_1 * X_h_2 * P_h * C_h_1 * C_h_2 * R_h

    # ----------------------------- The Newton loop ----------------------------

    def newton_solve(sln):

        # ---- Modify source terms to preserve chemical potential nullspace ----
        g_1_discrete_bc = project_util(g_1, W_h_1, bcs=[DirichletBC(W_h_1, g_1, (1, 2, 3, 4))])
        g_2_discrete_bc = project_util(g_2, W_h_2, bcs=[DirichletBC(W_h_2, g_2, (1, 2, 3, 4))])

        Sm_h = FunctionSpace(mesh, "DG" if mesh_type == "triangle" else "DQ", k + 2)

        r_1_discrete = project_util(r_1, Sm_h)
        r_2_discrete = project_util(r_2, Sm_h)

        r_1_delta = Constant((((1.0 / M_1) * assemble(inner(g_1_discrete_bc, nml) * ds(degree=deg_max))) \
                             - assemble(r_1_discrete * dx(degree=deg_max))) / vol)
        r_2_delta = Constant((((1.0 / M_2) * assemble(inner(g_2_discrete_bc, nml) * ds(degree=deg_max))) \
                             - assemble(r_2_discrete * dx(degree=deg_max))) / vol)

        r_1_discrete += r_1_delta
        r_2_discrete += r_2_delta

        # ------------------------- The discrete forms -------------------------

        # Trial and test functions
        mm_1, mm_2, v, mu_1, mu_2, p, c_1, c_2, rho_inv = split(sln)
        u_1, u_2, u, w_1, w_2, q, y_1, y_2, r = TestFunctions(Z_h)

        # The viscous terms
        A_visc = 2.0 * eta * inner(sym(grad(v)), sym(grad(u))) * dx(degree=deg_max)
        A_visc += lame * inner(div(v), div(u)) * dx(degree=deg_max)

        # The OSM terms
        A_osm = (RT / ((c_1 + c_2) * D_12)) * ((c_2 / (M_1 * M_1 * c_1)) * inner(mm_1, u_1) \
                                          + (c_1 / (M_2 * M_2 * c_2)) * inner(mm_2, u_2) \
                                          - (1.0 / (M_1 * M_2)) * (inner(mm_1, u_2) + inner(mm_2, u_1))) * dx(degree=deg_max)
        A_osm += gamma * inner(v - (rho_inv * (mm_1 + mm_2)), u - (rho_inv * (u_1 + u_2))) * dx(degree=deg_max)

        # The diffusion driving force terms (and their transposes)
        B_blf = (inner(p, div(rho_inv * (u_1 + u_2))) - inner(p, div(u))) * dx(degree=deg_max)
        B_blf -= ((1.0 / M_1) * inner(mu_1, div(u_1)) + (1.0 / M_2) * inner(mu_2, div(u_2))) * dx(degree=deg_max)

        BT_blf = (inner(q, div(rho_inv * (mm_1 + mm_2))) - inner(q, div(v))) * dx(degree=deg_max)
        BT_blf -= ((1.0 / M_1) * inner(w_1, div(mm_1)) + (1.0 / M_2) * inner(w_2, div(mm_2))) * dx(degree=deg_max)

        # The total residual
        tot_res = A_visc + A_osm + B_blf + BT_blf

        # The concentration and density terms
        tot_res += (inner(c_1, y_1) + inner(c_2, y_2)) * dx(degree=deg_max)
        tot_res -= (inner(ideal_gas(mu_1), y_1) + inner(ideal_gas(mu_2), y_2)) * dx(degree=deg_max)

        tot_res += inner(1.0 / rho_inv, r) * dx(degree=deg_max)
        tot_res -= inner((M_1 * c_1) + (M_2 * c_2), r) * dx(degree=deg_max)

        # The density consistency terms
        tot_res -= p * inner((rho_inv * (u_1 + u_2)) - u, nml) * ds(degree=deg_max)
        tot_res -= q * inner((rho_inv * (mm_1 + mm_2)) - v, nml) * ds(degree=deg_max)

        # The forcing terms
        tot_res -= (inner(f * ((M_1 * c_1) + (M_2 * c_2)), u) - inner(w_1, r_1_discrete) - inner(w_2, r_2_discrete)) * dx(degree=deg_max)

        # -------------------------- Specify the BCs ---------------------------

        # The BCs that are supplied to Firedrake to perform Newton's method
        newton_bcs = [DirichletBC(Z_h.sub(0), g_1, (1, 2, 3, 4)),
                      DirichletBC(Z_h.sub(1), g_2, (1, 2, 3, 4)),
                      DirichletBC(Z_h.sub(2), g_v, (1, 2, 3, 4)),
                      FixAtPointBC(Z_h.sub(5), p_ref, None, bc_point_ref)]

        # Auxiliary point BCs on the chemical potentials to remove the nullspaces
        auxiliary_point_bcs = [FixAtPointBC(Z_h.sub(3), Constant(1.0), None, bc_point_ref),
                               FixAtPointBC(Z_h.sub(4), Constant(1.0), None, bc_point_ref)]

        # ------------------ Configure Newton solver ---------------------------

        newton_solver_parameters = {"snes_type" : "newtonls",
                                    "snes_monitor" : "",
                                    "snes_converged_reason" : "",
                                    "snes_linesearch_type" : "basic",
                                    "snes_atol" : 0.0,          # Don't let PETSc do the convergence test
                                    "snes_stol" : 0.0,          # Don't let PETSc do the convergence test
                                    "snes_rtol" : 0.0,          # Don't let PETSc do the convergence test
                                    "ksp_type" : "gmres",
                                    "ksp_max_it" : 3,
                                    "ksp_convergence_test" : "skip",
                                    "pc_type" : "lu",
                                    "pc_factor_mat_solver_type" : "mumps",
                                    "mat_mumps_icntl_14" : 120,
                                    "mat_mumps_icntl_24" : 0,   # No null pivot detection
                                    "ksp_monitor" : ""}

        PETSc.Sys.Print("Solve Newton system (Refinement %d, N_mesh = %d)" % (i, N_mesh), flush=True)

        # Insert rows of the identity (corresponding to auxiliary_point_bcs) into the Jacobian
        def post_jacobian(sln_current_vec, J_current_mat):

            i1 = auxiliary_point_bcs[0].dof_index_in_mixed_space(Z_h, 3)
            i2 = auxiliary_point_bcs[1].dof_index_in_mixed_space(Z_h, 4)
            J_current_mat.zeroRows([i1, i2], diag=1.0)

        # Modify entries of the residual (corresponding to auxiliary_point_bcs) for enforcing the concentration constraints
        def post_function(sln_current_vec, residual_current_vec):

            sln_current_func = Function(Z_h)
            with sln_current_func.dat.vec_wo as sln_current_func_data:
                    sln_current_vec.copy(sln_current_func_data)

            c_1_current = sln_current_func.subfunctions[6]
            c_2_current = sln_current_func.subfunctions[7]

            new_val_1 = float(assemble(c_1_current * dx(degree=deg_max)) - c_1_total)
            new_val_2 = float(assemble(c_2_current * dx(degree=deg_max)) - c_2_total)

            i1 = auxiliary_point_bcs[0].dof_index_in_mixed_space(Z_h, 3)
            i2 = auxiliary_point_bcs[1].dof_index_in_mixed_space(Z_h, 4)

            residual_current_vec.assemblyBegin()
            residual_current_vec.setValues([i1, i2], [new_val_1, new_val_2])
            residual_current_vec.assemblyEnd()

        # Define the nonlinear problem and solver
        newton_problem = NonlinearVariationalProblem(tot_res, sln, bcs=newton_bcs)
        newton_solver = NonlinearVariationalSolver(newton_problem, solver_parameters=newton_solver_parameters,
                                                   post_jacobian_callback=post_jacobian,
                                                   post_function_callback=post_function)

        # ------------------ Configure Newton monitor --------------------------

        if Z_h.mesh().comm.rank == 0 and i == 0:
            try:
                shutil.rmtree("newton_2d_no_cont_out")
            except:
                pass

        global newton_iter
        newton_iter = 0

        global newton_outfile_ns, newton_outfile_nu
        newton_outfile_ns = VTKFile("newton_2d_no_cont_out/refinement_%d/ns.pvd" % i)
        newton_outfile_nu = VTKFile("newton_2d_no_cont_out/refinement_%d/nu.pvd" % i)

        def newton_monitor(snes, its, nm):

            global newton_iter
            global newton_outfile_ns, newton_outfile_nu

            # Get the Newton solution and update
            newton_sln = Function(Z_h)
            newton_update = Function(Z_h)

            with newton_sln.dat.vec_wo as newton_sln_data:
                snes.getSolution().copy(newton_sln_data)

            with newton_update.dat.vec_wo as newton_update_data:
                snes.getSolutionUpdate().copy(newton_update_data)

            mm_1_ns, mm_2_ns, v_ns, mu_1_ns, mu_2_ns, p_ns, c_1_ns, c_2_ns, rho_inv_ns = newton_sln.subfunctions
            mm_1_nu, mm_2_nu, v_nu, mu_1_nu, mu_2_nu, p_nu, c_1_nu, c_2_nu, rho_inv_nu = newton_update.subfunctions

            # Save to file
            VSV = VectorFunctionSpace(mesh, "DG" if mesh_type == "triangle" else "DQ", k)
            VSS = FunctionSpace(mesh, "DG" if mesh_type == "triangle" else "DQ", k - 1)

            mm_1_ns_save = Function(VSV, name="mm_1")
            mm_2_ns_save = Function(VSV, name="mm_2")
            v_ns_save = Function(VSV, name="v")
            mu_1_ns_save = Function(VSS, name="mu_1")
            mu_2_ns_save = Function(VSS, name="mu_2")
            p_ns_save = Function(VSS, name="p")
            c_1_ns_save = Function(VSS, name="c_1")
            c_2_ns_save = Function(VSS, name="c_2")
            rho_inv_ns_save = Function(VSS, name="rho_inv")

            mm_1_nu_save = Function(VSV, name="mm_1")
            mm_2_nu_save = Function(VSV, name="mm_2")
            v_nu_save = Function(VSV, name="v")
            mu_1_nu_save = Function(VSS, name="mu_1")
            mu_2_nu_save = Function(VSS, name="mu_2")
            p_nu_save = Function(VSS, name="p")
            c_1_nu_save = Function(VSS, name="c_1")
            c_2_nu_save = Function(VSS, name="c_2")
            rho_inv_nu_save = Function(VSS, name="rho_inv")

            mm_1_ns_save.assign(project_util(mm_1_ns, VSV))
            mm_2_ns_save.assign(project_util(mm_2_ns, VSV))
            v_ns_save.assign(project_util(v_ns, VSV))
            mu_1_ns_save.assign(project_util(mu_1_ns, VSS))
            mu_2_ns_save.assign(project_util(mu_2_ns, VSS))
            p_ns_save.assign(project_util(p_ns, VSS))
            c_1_ns_save.assign(project_util(c_1_ns, VSS))
            c_2_ns_save.assign(project_util(c_2_ns, VSS))
            rho_inv_ns_save.assign(project_util(rho_inv_ns, VSS))

            mm_1_nu_save.assign(project_util(mm_1_nu, VSV))
            mm_2_nu_save.assign(project_util(mm_2_nu, VSV))
            v_nu_save.assign(project_util(v_nu, VSV))
            mu_1_nu_save.assign(project_util(mu_1_nu, VSS))
            mu_2_nu_save.assign(project_util(mu_2_nu, VSS))
            p_nu_save.assign(project_util(p_nu, VSS))
            c_1_nu_save.assign(project_util(c_1_nu, VSS))
            c_2_nu_save.assign(project_util(c_2_nu, VSS))
            rho_inv_nu_save.assign(project_util(rho_inv_nu, VSS))

            newton_outfile_ns.write(mm_1_ns_save, mm_2_ns_save, v_ns_save,
                                    mu_1_ns_save, mu_2_ns_save, p_ns_save, c_1_ns_save, c_2_ns_save,
                                    rho_inv_ns_save, time=newton_iter)
            newton_outfile_nu.write(mm_1_nu_save, mm_2_nu_save, v_nu_save,
                                    mu_1_nu_save, mu_2_nu_save, p_nu_save, c_1_nu_save, c_2_nu_save,
                                    rho_inv_nu_save, time=newton_iter)

            newton_iter += 1

        # ----- Configure SNES convergence test (also does Woodbury update) -----

        def convergence_test_woodbury(snes, its, nm):

            # Do nothing on the first iteration (i.e. before doing any Newton steps)
            if its == 0:
                return None

            # Carry out the Woodbury update
            i1 = auxiliary_point_bcs[0].dof_index_in_mixed_space(Z_h, 3)
            i2 = auxiliary_point_bcs[1].dof_index_in_mixed_space(Z_h, 4)

            u0_vec = snes.getSolutionUpdate()   # Careful: this gives (old solution - new solution)

            e1_vec = Z_h.dof_dset.layout_vec.copy()
            e2_vec = Z_h.dof_dset.layout_vec.copy()

            e1_vec.assemblyBegin()
            e2_vec.assemblyBegin()

            e1_vec.zeroEntries()
            e2_vec.zeroEntries()

            e1_vec.setValue(i1, 1.0)
            e2_vec.setValue(i2, 1.0)

            e1_vec.assemblyEnd()
            e2_vec.assemblyEnd()

            w1_vec = Z_h.dof_dset.layout_vec.copy()
            w2_vec = Z_h.dof_dset.layout_vec.copy()

            snes.getKSP().solve(e1_vec, w1_vec)
            snes.getKSP().solve(e2_vec, w2_vec)

            w1_norm = w1_vec.norm(norm_type=petsc4py.PETSc.NormType.N2)
            w2_norm = w2_vec.norm(norm_type=petsc4py.PETSc.NormType.N2)

            # TODO: These norms below can be very large (e.g. 10^8), this should be investigated
            PETSc.Sys.Print("Norm of w_i vectors: w_1 %.5e, w_2 %.5e" % (w1_norm, w2_norm))

            def v_mat_action(vec):

                vec_func = Function(Z_h)
                with vec_func.dat.vec_wo as vec_func_data:
                    vec.copy(vec_func_data)

                vec_func_mu_1 = vec_func.subfunctions[3]
                vec_func_mu_2 = vec_func.subfunctions[4]
                vec_func_c_1 = vec_func.subfunctions[6]
                vec_func_c_2 = vec_func.subfunctions[7]

                vec_func_mu_1_bc_val = auxiliary_point_bcs[0].eval_at_point(vec_func_mu_1)
                vec_func_mu_2_bc_val = auxiliary_point_bcs[1].eval_at_point(vec_func_mu_2)

                vec_func_c_1_mean = float(assemble(vec_func_c_1 * dx(degree=deg_max)))
                vec_func_c_2_mean = float(assemble(vec_func_c_2 * dx(degree=deg_max)))

                return np.array([vec_func_c_1_mean - vec_func_mu_1_bc_val,
                                 vec_func_c_2_mean - vec_func_mu_2_bc_val])

            v_u0 = v_mat_action(u0_vec)
            vw_mat = np.transpose(np.array([v_mat_action(w1_vec), v_mat_action(w2_vec)]))

            I_vw_mat = np.identity(2) + vw_mat
            alpha = npla.solve(I_vw_mat, v_u0)

            cond = npla.cond(I_vw_mat)
            PETSc.Sys.Print("Condition number of the Woodbury matrix is %.2e" % cond)

            new_sln_vec = snes.getSolution()
            new_sln_vec.axpy(alpha[0], w1_vec)
            new_sln_vec.axpy(alpha[1], w2_vec)

            u0_vec.axpy((-1.0) * alpha[0], w1_vec)
            u0_vec.axpy((-1.0) * alpha[1], w2_vec)

            # Check that the point BC and mean constraints were enforced
            new_sln = Function(Z_h)
            with new_sln.dat.vec_wo as new_sln_data:
                new_sln_vec.copy(new_sln_data)

            new_sln_p = new_sln.subfunctions[5]
            new_sln_c_1 = new_sln.subfunctions[6]
            new_sln_c_2 = new_sln.subfunctions[7]

            newton_bcs[-1].assert_is_enforced(new_sln_p)
            assert(float(abs(assemble(new_sln_c_1 * dx(degree=deg_max)) - c_1_total)) < 1e-8)
            assert(float(abs(assemble(new_sln_c_2 * dx(degree=deg_max)) - c_2_total)) < 1e-8)

            # Call the custom Newton monitor
            newton_monitor(snes, its, nm)

            # Compute the true post-Woodbury residual norm (it differs from the PETSc SNES residual norm)
            f_vec = Z_h.dof_dset.layout_vec.copy()
            snes.computeFunction(new_sln_vec, f_vec)

            f_vec.assemblyBegin()
            f_vec.setValues([i1, i2], [0.0, 0.0])   # Zero the entries corresponding to concentration constraint
            f_vec.assemblyEnd()

            f_norm = f_vec.norm(norm_type=petsc4py.PETSc.NormType.N2)
            PETSc.Sys.Print("True post-Woodbury residual norm: %.5e" % f_norm)

            # Decide if nonlinear convergence has been achieved
            if f_norm <= newton_atol:
                PETSc.Sys.Print("Newton solver has converged: sufficiently small residual norm")
                return petsc4py.PETSc.SNES.ConvergedReason.CONVERGED_FNORM_ABS
            elif snes.getIterationNumber() > newton_max_it:
                return petsc4py.PETSc.SNES.ConvergedReason.DIVERGED_MAX_IT
            else:
                return None

        newton_solver.snes.setConvergenceTest(convergence_test_woodbury)

        # ----------------------- Do the Newton solve --------------------------

        # The main Newton solve
        newton_solver.solve()
        PETSc.Sys.Print("Done Newton system (Refinement %d, N_mesh = %d) \n" % (i, N_mesh), flush=True)

        # Check (again) that the point BC and mean constraints were enforced
        mm_1, mm_2, v, mu_1, mu_2, p, c_1, c_2, rho_inv = sln.subfunctions

        newton_bcs[-1].assert_is_enforced(p)
        assert(float(abs(assemble(c_1 * dx(degree=deg_max)) - c_1_total)) < 1e-8)
        assert(float(abs(assemble(c_2 * dx(degree=deg_max)) - c_2_total)) < 1e-8)

        # Shift the pressure by a constant so that it has the same mean as the exact pressure
        p += Constant(exact_p_mean - assemble(p * dx(degree=deg_max)))

    # -------------------------- Build the initial guess -----------------------
    sln = Function(Z_h)

    mm_1, mm_2, v, mu_1, mu_2, p, c_1, c_2, rho_inv = sln.subfunctions

    project_bcs = [DirichletBC(W_h_1, g_1, (1, 2, 3, 4)),
                   DirichletBC(W_h_2, g_2, (1, 2, 3, 4)),
                   DirichletBC(V_h, g_v, (1, 2, 3, 4)),
                   FixAtPointBC(P_h, p_ref, None, bc_point_ref)]

    mm_1.assign(project_util(mm_1_ms, W_h_1, bcs=project_bcs[0]))
    mm_2.assign(project_util(mm_2_ms, W_h_2, bcs=project_bcs[1]))
    v.assign(project_util(v_ms, V_h, bcs=project_bcs[2]))
    mu_1.assign(project_util(mu_1_ms, X_h_1))
    mu_2.assign(project_util(mu_2_ms, X_h_2))
    p.assign(project_util(p_ms, P_h, bcs=project_bcs[3]))
    c_1.assign(project_util(c_1_ms, C_h_1))
    c_2.assign(project_util(c_2_ms, C_h_2))
    rho_inv.assign(project_util(1 / rho_ms, R_h))

    # ------------------------- Doing the Newton solve -------------------------
    newton_solve(sln)

    # ----------------------------- Compute the errors -------------------------
    mm_1, mm_2, v, mu_1, mu_2, p, c_1, c_2, rho_inv = sln.subfunctions

    # Generate Paraview plots of the errors?
    plot_errors = False
    if plot_errors:

        ErrVSV = VectorFunctionSpace(mesh, "DG" if mesh_type == "triangle" else "DQ", k + 3)
        ErrVSS = FunctionSpace(mesh, "DG" if mesh_type == "triangle" else "DQ", k + 2)

        mm_1_err = Function(ErrVSV, name="mm_1")
        mm_2_err = Function(ErrVSV, name="mm_2")
        v_err = Function(ErrVSV, name="v")
        mu_1_err = Function(ErrVSS, name="mu_1")
        mu_2_err = Function(ErrVSS, name="mu_2")
        p_err = Function(ErrVSS, name="p")
        mass_avg_err = Function(ErrVSV, name="mass_avg")
        c_1_err = Function(ErrVSS, name="c_1")
        c_2_err = Function(ErrVSS, name="c_2")
        rho_inv_err = Function(ErrVSS, name="rho_inv")

        mm_1_err.assign(project_util(mm_1 - mm_1_ms, ErrVSV))
        mm_2_err.assign(project_util(mm_2 - mm_2_ms, ErrVSV))
        v_err.assign(project_util(v - v_ms, ErrVSV))
        mu_1_err.assign(project_util(mu_1 - mu_1_ms, ErrVSS))
        mu_2_err.assign(project_util(mu_2 - mu_2_ms, ErrVSS))
        p_err.assign(project_util(p - p_ms, ErrVSS))
        mass_avg_err.assign(project_util(v - rho_inv * (mm_1 + mm_2), ErrVSV))
        c_1_err.assign(project_util(c_1 - c_1_ms, ErrVSS))
        c_2_err.assign(project_util(c_2 - c_2_ms, ErrVSS))
        rho_inv_err.assign(project_util(rho_inv - (1 / rho_ms), ErrVSS))

        VTKFile("newton_2d_no_cont_out/refinement_%d/errs.pvd" % i).write(mm_1_err, mm_2_err, v_err, mu_1_err, mu_2_err, \
                                                                          p_err, mass_avg_err, c_1_err, c_2_err, rho_inv_err)

    # Compute the errors for the unknowns
    mu_1_err = l2_dist(mu_1_ms, mu_1)
    mu_2_err = l2_dist(mu_2_ms, mu_2)

    grad_mu_1_err = l2_dist(grad(mu_1_ms), grad(mu_1))
    grad_mu_2_err = l2_dist(grad(mu_2_ms), grad(mu_2))

    p_err = l2_dist(p_ms, p)
    grad_p_err = l2_dist(grad(p_ms), grad(p))

    mm_1_err = l2_dist(mm_1_ms, mm_1)
    mm_2_err = l2_dist(mm_2_ms, mm_2)

    div_mm_1_err = l2_dist(div(mm_1_ms), div(mm_1))
    div_mm_2_err = l2_dist(div(mm_2_ms), div(mm_2))

    v_err = l2_dist(v_ms, v)
    grad_v_err = l2_dist(grad(v_ms), grad(v))

    mass_avg_err = l2_dist(v, rho_inv * (mm_1 + mm_2))

    c_1_err = l2_dist(c_1_ms, c_1)
    c_2_err = l2_dist(c_2_ms, c_2)

    rho_inv_err = l2_dist(1 / rho_ms, rho_inv)

    # Store the absolute and relative errors
    errs[i, :, 0] = np.array([mu_1_err, mu_2_err, grad_mu_1_err, grad_mu_2_err,
                              p_err, grad_p_err,
                              mm_1_err, mm_2_err, div_mm_1_err, div_mm_2_err,
                              v_err, grad_v_err, mass_avg_err,
                              c_1_err, c_2_err, rho_inv_err])

    norms = np.array([l2_norm(mu_1_ms), l2_norm(mu_2_ms), l2_norm(grad(mu_1_ms)), l2_norm(grad(mu_2_ms)),
                      l2_norm(p_ms), l2_norm(grad(p_ms)),
                      l2_norm(mm_1_ms), l2_norm(mm_2_ms), l2_norm(div(mm_1_ms)), l2_norm(div(mm_2_ms)),
                      l2_norm(v_ms), l2_norm(grad(v_ms)), l2_norm(v_ms),
                      l2_norm(c_1_ms), l2_norm(c_2_ms), l2_norm(1 / rho_ms)])

    errs[i, :, 1] = np.multiply(errs[i, :, 0], 1.0 / norms)

# ----------------------------- Display the errors and rates -------------------

# Save the errors to numpy format?
save_errors = False
if save_errors:
    np.save("newton_2d_no_cont_out/errs.npy", errs, allow_pickle=False)

# The name of the fields
errs_names = ["mu_1", "mu_2", "grad_mu_1", "grad_mu_2",
               "p", "grad_p",
               "mm_1", "mm_2", "div_mm_1", "div_mm_2",
               "v", "grad_v", "mass_avg",
               "c_1", "c_2", "rho_inv"]

# The associated theoretical convergence rates
rates_theory = [str(k), str(k), "-", "-",
                str(k), "-",
                str(k), str(k), str(k), str(k),
                str(k), str(k), str(k),
                "-", "-", "-"]

# The associated optimal convergence rates
rates_optimal = [str(k), str(k), str(k - 1), str(k - 1),
                 str(k), str(k - 1),
                 str(k + 1), str(k + 1), str(k), str(k),
                 str(k + 1), str(k), str(k + 1),
                 str(k), str(k), str(k)]

# Generate the table
table_headers = ["Field", "Relative Errors", "Absolute Errors", "Rates", "Theoretical\nRates", "Optimal\nRates"]
table_data = []

for k in range(0, num_err_fields):

    row = []
    row.append(errs_names[k])

    relative_errs_str = ""
    absolute_errs_str = ""
    rates_str = "-"
    theory_str = "-"
    optimal_str = "-"

    for i in range(0, n_loops):
        relative_errs_str += "%.2e" % errs[i, k, 1]
        absolute_errs_str += "%.2e" % errs[i, k, 0]

        if i > 0:
            rates_str += "%.2f" % (np.log(errs[i - 1, k, 0] / errs[i, k, 0]) / np.log(2))
            theory_str += rates_theory[k]
            optimal_str += rates_optimal[k]

        if i < n_loops - 1:
            relative_errs_str += "\n"
            absolute_errs_str += "\n"
            rates_str += "\n"
            theory_str += "\n"
            optimal_str += "\n"

    row.append(relative_errs_str)
    row.append(absolute_errs_str)
    row.append(rates_str)
    row.append(theory_str)
    row.append(optimal_str)

    table_data.append(row)

# Show the table
PETSc.Sys.Print(tabulate(table_data, headers=table_headers, tablefmt="fancy_grid"), flush=True)

# --------------------------- Print to LaTeX table format ----------------------
latex = False

errs_names_latex = ["\\mu_1", "\\mu_2", "\\nabla \\mu_1", "\\nabla \\mu_2",
               "p", "\\nabla p",
               "\\tilde{v}_1", "\\tilde{v}_2", "\\div \\tilde{v}_1", "\\div \\tilde{v}_2",
               "v", "\\nabla v", "v - \\sum_i \\omega_i v_i",
               "c_1", "c_2", "\\rho^{-1}"]

if latex:
    for k in range(0, num_err_fields):

        relative_errs_str = "& \\makecell{"
        absolute_errs_str = "& \\makecell{"
        rates_str = "& \\makecell{-"
        theory_str = "& \\makecell{-"
        optimal_str = "& \\makecell{-"

        for i in range(0, n_loops):
            relative_errs_str += "%.2e" % errs[i, k, 1]
            absolute_errs_str += "%.2e" % errs[i, k, 0]

            if i > 0:
                rates_str += "%.2f" % (np.log(errs[i - 1, k, 0] / errs[i, k, 0]) / np.log(2))
                theory_str += rates_theory[k]
                optimal_str += rates_optimal[k]

            if i < n_loops - 1:
                relative_errs_str += " \\\\ "
                absolute_errs_str += " \\\\ "
                rates_str += " \\\\ "
                theory_str += " \\\\ "
                optimal_str += " \\\\ "

        relative_errs_str += "}"
        absolute_errs_str += "}"
        rates_str += "}"
        theory_str += "}"
        optimal_str += "}"

        out = "% \n${0}$ \n{1} \n{2} \n{3} \n{4} \n{5} \\\\ \\hline \n"
        out = out.format(errs_names_latex[k], relative_errs_str, absolute_errs_str, rates_str, theory_str, optimal_str)
        PETSc.Sys.Print(out, flush=True)
