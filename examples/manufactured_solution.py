from firedrake import *
import firedrake.utils as firedrake_utils
from firedrake.petsc import PETSc
import petsc4py
import numpy as np
import numpy.linalg as npla
from tabulate import tabulate

# -------------- Some key parameters relating to the numerics -----------------

# The tests in the paper use the following parameter values:
# (i) [2D linearized test case]: d=2, k=4, mesh_type=tet, picard_linearized=True, N_mesh_initial=8, n_loops=5
# (ii) [2D non-linear test case]: d=2, k=4, mesh_type=tet, picard_linearized=False, N_mesh_initial=8, n_loops=5
# (iii) [3D linearized test case]: d=3, k=4, mesh_type=hex, picard_linearized=True, N_mesh_initial=2, n_loops=4
# (iv) [3D non-linear test case]: d=3, k=4, mesh_type=hex, picard_linearized=False, N_mesh_initial=2, n_loops=4
# ... and deg_max=15, density_consistency=True, use_grad_rho_inv_exact=False for all tests

d = 2                           # The spatial dimension
assert d in [2, 3]

k = 4                           # The polynomial degree (for the flux spaces)
deg_max = 15                    # Maximum quadrature degree

mesh_type = "tet"               # What mesh type to use (in 2D "tet" means triangle and "hex" means quad)
assert mesh_type in ["tet", "hex"]

picard_linearized = True        # Solve the Picard linearized problem or the full nonlinear problem?

density_consistency = True      # Include density consistency terms (only for the full nonlinear problem)?

use_grad_rho_inv_exact = False  # Replace occurences of grad(rho_inv_h) with grad(rho_inv) in the nonlinear discretization,
                                # where rho_inv_h is the discrete density reciprocal and rho_inv the exact density reciprocal

N_mesh_initial = 8              # Initial number of mesh elements in each spatial direction
n_loops = 5                     # Number of mesh refinements to be done

# -------------- Looping over meshes to get convergence rates -----------------

num_err_fields = 16
errs = np.empty((n_loops, num_err_fields, 2))   # The errors to be kept track of

for i in range(0, n_loops):

    # ----------------------------- The computational mesh --------------------

    N_mesh = N_mesh_initial * (2 ** i)

    if d == 2:
        mesh = UnitSquareMesh(N_mesh, N_mesh, quadrilateral=(mesh_type == "hex"))
        bc_markers = (1, 2, 3, 4)
        ds = ds(mesh, degree=deg_max)
        dx = dx(mesh, degree=deg_max)

    elif d == 3:
        if mesh_type == "tet":
            mesh = UnitCubeMesh(N_mesh, N_mesh, N_mesh)
            bc_markers = (1, 2, 3, 4, 5, 6)
            ds = ds(mesh, degree=deg_max) 
            dx = dx(mesh, degree=deg_max)

        elif mesh_type == "hex":
            mesh_2d = UnitSquareMesh(N_mesh, N_mesh, quadrilateral=True)
            mesh = ExtrudedMesh(mesh_2d, N_mesh)
            bc_markers = (1, 2, 3, 4, "top", "bottom")
            ds = ds_t(mesh, degree=deg_max) + ds_b(mesh, degree=deg_max) + ds_v(mesh, degree=deg_max)
            dx = dx(mesh, degree=deg_max)

    vol = assemble(1.0*dx)  # Volume of the mesh

    # Utility functions for L^2 norm and distance
    def l2_norm(f):
        return np.sqrt(assemble(inner(f, f) * dx))

    def l2_dist(f, g):
        return l2_norm(f - g)

    # Utility class for fixing a (possibly DG) field at a node, for the auxiliary constraints
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

    # ----------------------------- The problem parameters --------------------

    # Physical parameters

    # IMPORTANT: Molar masses must be one for the manufactured solution
    M_1 = Constant(1)           # Molar mass of species 1
    M_2 = Constant(1)           # Molar mass of species 2

    # IMPORTANT: RT must be one for the manufactured solution
    RT = Constant(1)            # Ideal gas constant times the ambient temperature

    # IMPORTANT: Need to define D_12 in this special way for the manufactured solution
    D_1 = Constant(0.5)         # Parameter for defining the Stefan-Maxwell diffusivity
    D_2 = Constant(2.0)         # Parameter for defining the Stefan-Maxwell diffusivity
    D_12 = D_1 * D_2            # Stefan-Maxwell diffusivity (= D_21)

    zeta = Constant(1e-1)                       # Bulk viscosity for Stokes
    eta = Constant(1e-1)                        # Shear viscosity for Stokes
    lame = (zeta - ((2.0 * eta) / d))           # Lame parameter

    # Other parameters
    gamma = Constant(1e1)                       # The augmentation parameter

    newton_atol = 1e-10                         # Newton absolute tolerance
    newton_max_it = 10                          # Newton maximum iterations

    # ----------------- Thermodynamic constitutive relations ------------------

    # Ideal gas law for chemical potentials
    def mu_relation(x_1, x_2, p):
        mu_1 = RT * ln(x_1 * p)
        mu_2 = RT * ln(x_2 * p)
        return (mu_1, mu_2)

    # Volumetric equation of state
    def conc_relation(x_1, x_2, p):
        x_1_nm = x_1 / (x_1 + x_2)
        x_2_nm = x_2 / (x_1 + x_2)

        c_tot = p / RT
        c_1 = x_1_nm * c_tot
        c_2 = x_2_nm * c_tot

        return (c_tot, c_1, c_2)

    # ----------------------------- The manufactured solution -----------------

    x = SpatialCoordinate(mesh)

    nml = FacetNormal(mesh)
    Id = Identity(d)

    if d == 2:
        g_ms = sin(pi * x[0]) * sin(pi * x[1])
    elif d == 3:
        g_ms = sin(pi * x[0]) * sin(pi * x[1]) * sin(pi * x[2])

    mu_1_ms = g_ms / D_1                # Chemical potential of species 1
    mu_2_ms = g_ms / D_2                # Chemical potential of species 2

    c_1_ms = exp(mu_1_ms)               # Concentration of species 1
    c_2_ms = exp(mu_2_ms)               # Concentration of species 2

    c_T_ms = c_1_ms + c_2_ms                    # Total concentration
    rho_ms = (M_1 * c_1_ms) + (M_2 * c_2_ms)    # Density
    rho_inv_ms = 1 / rho_ms                     # Density reciprocal

    x_1_ms = c_1_ms / c_T_ms                    # Mole fraction of species 1
    x_2_ms = c_2_ms / c_T_ms                    # Mole fraction of species 2

    omega_1_ms = (M_1 * c_1_ms) / rho_ms        # Mass fraction of species 1
    omega_2_ms = (M_2 * c_2_ms) / rho_ms        # Mass fraction of species 2

    v_1_ms = D_1 * grad(g_ms)                               # Velocity of species 1
    v_2_ms = D_2 * grad(g_ms)                               # Velocity of species 2
    v_ms = (omega_1_ms * v_1_ms) + (omega_2_ms * v_2_ms)    # Mass average velocity

    mm_1_ms = M_1 * c_1_ms * v_1_ms                         # Mass flux of species 1
    mm_2_ms = M_2 * c_2_ms * v_2_ms                         # Mass flux of species 2

    epsilon_v_ms = sym(grad(v_ms))                                  # Strain rate
    tau_ms = (2.0 * eta * epsilon_v_ms) + \
             lame * tr(epsilon_v_ms) * Id                           # Viscous stress
    p_ms = c_T_ms                                                   # Pressure

    f_ms = (grad(p_ms) - div(tau_ms))                               # Stokes forcing term (times the density)

    # ----------------------------- The problem data --------------------------

    f = f_ms / rho_ms                           # Stokes forcing term

    r_1 = div(c_1_ms * v_1_ms)                  # Volumetric generation/depletion rate of species 1
    r_2 = div(c_2_ms * v_2_ms)                  # Volumetric generation/depletion rate of species 2

    g_v = v_ms                                  # Boundary condition for the barycentric velocity

    T_1 = Constant(1.0)                         # To exactly satisfy the compatibility condition at the discrete level
    T_2 = Constant(1.0)                         # To exactly satisfy the compatibility condition at the discrete level

    g_1 = T_1 * mm_1_ms                         # (Normal) boundary condition for species 1 mass flux
    g_2 = T_2 * mm_2_ms                         # (Normal) boundary condition for species 2 mass flux

    # The boundary point that is used for the auxiliary constraints
    bc_point_ref = Constant([0.0, 0.0]) if d == 2 else Constant([0.0, 0.0, 0.0])

    # The total concentrations and mole fraction integral, to be enforced using the Woodbury identity
    c_1_integral = Constant(assemble(c_1_ms * dx))
    c_2_integral = Constant(assemble(c_2_ms * dx))
    mfs_integral = 1.0 * vol

    # ----------------- Solver parameters used for projections ----------------

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

    # ----------------------------- The discrete spaces -----------------------

    if mesh_type == "tet":
        var = "equispaced"  # So that the chemical potentials and pressure have a DOF on bc_point_ref
        cell_type = triangle if d == 2 else tetrahedron
        W_h_1 = FunctionSpace(mesh, "RT", k)                                                    # Mass flux space for species 1
        W_h_2 = FunctionSpace(mesh, "RT", k)                                                    # Mass flux space for species 2
        V_h   = VectorFunctionSpace(mesh, "CG", k)                                              # Barycentric velocity space
        U_h_1 = FunctionSpace(mesh, FiniteElement("DG", cell_type, k - 1, variant=var))         # Chemical potential space for species 1
        U_h_2 = FunctionSpace(mesh, FiniteElement("DG", cell_type, k - 1, variant=var))         # Chemical potential space for species 2
        P_h   = FunctionSpace(mesh, "CG", k - 1)                                                # Pressure space
        X_h_1 = FunctionSpace(mesh, "DG", k - 1)                                                # Mole fraction space for species 1
        X_h_2 = FunctionSpace(mesh, "DG", k - 1)                                                # Mole fraction space for species 2
        R_h = FunctionSpace(mesh, "CG", k - 1)                                                  # Density reciprocal space

    elif mesh_type == "hex":
        var = "equispaced"  # So that the chemical potentials and pressure have a DOF on bc_point_ref
        cell_type = quadrilateral if d == 2 else hexahedron
        W_h_1 = FunctionSpace(mesh, "RTCF" if d == 2 else "NCF", k)                             # Mass flux space for species 1
        W_h_2 = FunctionSpace(mesh, "RTCF" if d == 2 else "NCF", k)                             # Mass flux space for species 2
        V_h   = VectorFunctionSpace(mesh, "CG", k)                                              # Barycentric velocity space
        U_h_1 = FunctionSpace(mesh, FiniteElement("DQ", cell_type, k - 1, variant=var))         # Chemical potential space for species 1
        U_h_2 = FunctionSpace(mesh, FiniteElement("DQ", cell_type, k - 1, variant=var))         # Chemical potential space for species 2
        P_h   = FunctionSpace(mesh, "CG", k - 1)                                                # Pressure space
        X_h_1 = FunctionSpace(mesh, "DQ", k - 1)                                                # Mole fraction space for species 1
        X_h_2 = FunctionSpace(mesh, "DQ", k - 1)                                                # Mole fraction space for species 2
        R_h = FunctionSpace(mesh, "CG", k - 1)                                                  # Density reciprocal space

    # -------------------- Solve the full nonlinear problem -------------------

    actually_evaluate = True    # A hack to prevent PETSc from evaluating the residual before we do the Woodbury update

    def solve_nonlinear():

        # The mixed function space for the full nonlinear problem
        Z_h = W_h_1 * W_h_2 * V_h * U_h_1 * U_h_2 * P_h * X_h_1 * X_h_2 * R_h

        # --------------------------- The Newton loop -------------------------

        def newton_solve(sln):

            # Modify T_1 and T_2 to exactly satisfy the compatibility condition at the discrete level
            cc_aux_J_1 = project_util(g_1, W_h_1, bcs=[DirichletBC(W_h_1, g_1, bc_markers)])
            cc_aux_J_2 = project_util(g_2, W_h_2, bcs=[DirichletBC(W_h_2, g_2, bc_markers)])

            Sm_h = FunctionSpace(mesh, "DG" if mesh_type == "tet" else "DQ", k + 2)
            r_1_discrete = project_util(r_1, Sm_h)
            r_2_discrete = project_util(r_2, Sm_h)

            T_1.assign(M_1 * assemble(r_1_discrete * dx) / assemble(inner(cc_aux_J_1, nml) * ds))
            T_2.assign(M_2 * assemble(r_2_discrete * dx) / assemble(inner(cc_aux_J_2, nml) * ds))

            PETSc.Sys.Print(f"Compatibility condition on fluxes - T_1 is: {float(T_1)}", flush=True)
            PETSc.Sys.Print(f"Compatibility condition on fluxes - T_2 is: {float(T_2)}", flush=True)

            # ----------------------- The discrete forms ----------------------

            # Trial and test functions
            mm_1, mm_2, v, mu_1, mu_2, p, x_1, x_2, rho_inv = split(sln)
            u_1, u_2, u, w_1, w_2, q, y_1, y_2, r = TestFunctions(Z_h)

            # Use the discrete density reciprocal gradient, or the exact density reciprocal gradient?
            grad_rho_inv = grad(rho_inv_ms) if use_grad_rho_inv_exact else grad(rho_inv)

            # The concentrations
            c_tot, c_1, c_2 = conc_relation(x_1, x_2, p)

            # The Stokes viscous terms
            A_visc = 2.0 * eta * inner(sym(grad(v)), sym(grad(u))) * dx
            A_visc += lame * inner(div(v), div(u)) * dx

            # The augmented Onsager transport matrix terms
            A_osm = (RT / ((c_1 + c_2) * D_12)) * ((c_2 / (M_1 * M_1 * c_1)) * inner(mm_1, u_1) \
                                            + (c_1 / (M_2 * M_2 * c_2)) * inner(mm_2, u_2) \
                                            - (1.0 / (M_1 * M_2)) * (inner(mm_1, u_2) + inner(mm_2, u_1))) * dx
            A_osm += gamma * inner(v - (rho_inv * (mm_1 + mm_2)), u - (rho_inv * (u_1 + u_2))) * dx

            # The diffusion driving force terms and Stokes pressure term
            B_blf = (inner(p, (rho_inv * div(u_1 + u_2)) + dot(grad_rho_inv, u_1 + u_2)) - inner(p, div(u))) * dx
            B_blf -= ((1.0 / M_1) * inner(mu_1, div(u_1)) + (1.0 / M_2) * inner(mu_2, div(u_2))) * dx

            # The div(mass-average constraint) and continuity equation terms
            BT_blf = (inner(q, (rho_inv * div(mm_1 + mm_2)) + dot(grad_rho_inv, mm_1 + mm_2)) - inner(q, div(v))) * dx
            BT_blf -= ((1.0 / M_1) * inner(w_1, div(mm_1)) + (1.0 / M_2) * inner(w_2, div(mm_2))) * dx

            # The total residual
            tot_res = A_visc + A_osm + B_blf + BT_blf

            # The thermodynamic constitutive relation and density reciprocal terms
            mu_1_cr, mu_2_cr = mu_relation(x_1, x_2, p)
            tot_res += (inner(mu_1 - mu_1_cr, y_1) + inner(mu_2 - mu_2_cr, y_2)) * dx

            tot_res += inner(1.0 / rho_inv, r) * dx
            tot_res -= inner((M_1 * c_1) + (M_2 * c_2), r) * dx

            # The density consistency terms
            if density_consistency:
                tot_res -= q * inner((rho_inv * (mm_1 + mm_2)) - v, nml) * ds

            # The forcing terms
            tot_res -= (inner(f * ((M_1 * c_1) + (M_2 * c_2)), u) - inner(w_1, r_1_discrete) - inner(w_2, r_2_discrete)) * dx

            # ----------------------- Specify the BCs -------------------------

            # The BCs on the mass fluxes and barycentric velocity
            newton_bcs = [DirichletBC(Z_h.sub(0), g_1, bc_markers),
                          DirichletBC(Z_h.sub(1), g_2, bc_markers),
                          DirichletBC(Z_h.sub(2), g_v, bc_markers)]

            # Auxiliary point BCs on the chemical potentials and pressure
            auxiliary_point_bcs = [FixAtPointBC(Z_h.sub(3), Constant(0.0), None, bc_point_ref),
                                   FixAtPointBC(Z_h.sub(4), Constant(0.0), None, bc_point_ref),
                                   FixAtPointBC(Z_h.sub(5), Constant(0.0), None, bc_point_ref)]

            # ------------------ Configure Newton solver ----------------------

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
                i3 = auxiliary_point_bcs[2].dof_index_in_mixed_space(Z_h, 5)

                J_current_mat.zeroRows([i1, i2, i3], diag=1.0)

            # Modify entries of the residual (corresponding to auxiliary_point_bcs) for enforcing the true constraints
            def post_function(sln_current_vec, residual_current_vec):

                global actually_evaluate
                if actually_evaluate:

                    # Compute the total concentration and mole fraction integrals, at the present solution
                    sln_current_func = Function(Z_h)
                    with sln_current_func.dat.vec_wo as sln_current_func_data:
                            sln_current_vec.copy(sln_current_func_data)

                    p_current = sln_current_func.subfunctions[5]
                    x_1_current = sln_current_func.subfunctions[6]
                    x_2_current = sln_current_func.subfunctions[7]
                    
                    c_tot_current, c_1_current, c_2_current = conc_relation(x_1_current, x_2_current, p_current)

                    new_val_c_1 = float(assemble(c_1_current * dx) - c_1_integral)
                    new_val_c_2 = float(assemble(c_2_current * dx) - c_2_integral)
                    new_val_mfs = float(assemble((x_1_current + x_2_current) * dx) - mfs_integral)

                    # Insert these values into the residual at the indices corresponding to the auxillary point BCs
                    i1 = auxiliary_point_bcs[0].dof_index_in_mixed_space(Z_h, 3)
                    i2 = auxiliary_point_bcs[1].dof_index_in_mixed_space(Z_h, 4)
                    i3 = auxiliary_point_bcs[2].dof_index_in_mixed_space(Z_h, 5)

                    residual_current_vec.assemblyBegin()
                    residual_current_vec.setValues([i1, i2, i3], [new_val_c_1, new_val_c_2, new_val_mfs])
                    residual_current_vec.assemblyEnd()

                    actually_evaluate = False

                else:
                    # Trick PETSc into thinking the residual is always 1e-12 (the value we use doesn't matter)
                    residual_current_vec.assemblyBegin()
                    residual_current_vec.zeroEntries()
                    residual_current_vec.setValue(1, 1e-12)
                    residual_current_vec.assemblyEnd()

            # Define the nonlinear problem and solver
            newton_problem = NonlinearVariationalProblem(tot_res, sln, bcs=newton_bcs)
            newton_solver = NonlinearVariationalSolver(newton_problem, solver_parameters=newton_solver_parameters,
                                                    post_jacobian_callback=post_jacobian,
                                                    post_function_callback=post_function)

            # -- Configure SNES convergence test (also does Woodbury update) --

            def convergence_test_woodbury(snes, its, nm):

                # Do nothing on the first iteration (i.e. before doing any Newton steps)
                if its == 0:
                    return None

                # --------------- Carry out the Woodbury update ---------------

                i1 = auxiliary_point_bcs[0].dof_index_in_mixed_space(Z_h, 3)
                i2 = auxiliary_point_bcs[1].dof_index_in_mixed_space(Z_h, 4)
                i3 = auxiliary_point_bcs[2].dof_index_in_mixed_space(Z_h, 5)

                u0_vec = snes.getSolutionUpdate()   # NOTE: this gives (old solution - new solution)

                e1_vec = Z_h.dof_dset.layout_vec.copy()
                e2_vec = Z_h.dof_dset.layout_vec.copy()
                e3_vec = Z_h.dof_dset.layout_vec.copy()

                e1_vec.assemblyBegin()
                e2_vec.assemblyBegin()
                e3_vec.assemblyBegin()

                e1_vec.zeroEntries()
                e2_vec.zeroEntries()
                e3_vec.zeroEntries()

                e1_vec.setValue(i1, 1.0)
                e2_vec.setValue(i2, 1.0)
                e3_vec.setValue(i3, 1.0)

                e1_vec.assemblyEnd()
                e2_vec.assemblyEnd()
                e3_vec.assemblyEnd()

                w1_vec = Z_h.dof_dset.layout_vec.copy()
                w2_vec = Z_h.dof_dset.layout_vec.copy()
                w3_vec = Z_h.dof_dset.layout_vec.copy()

                snes.getKSP().solve(e1_vec, w1_vec)
                snes.getKSP().solve(e2_vec, w2_vec)
                snes.getKSP().solve(e3_vec, w3_vec)

                w1_norm = w1_vec.norm(norm_type=petsc4py.PETSc.NormType.N2)
                w2_norm = w2_vec.norm(norm_type=petsc4py.PETSc.NormType.N2)
                w3_norm = w3_vec.norm(norm_type=petsc4py.PETSc.NormType.N2)

                PETSc.Sys.Print("Norm of w_i vectors: w_1 %.5e, w_2 %.5e, w_3 %.5e" % (w1_norm, w2_norm, w3_norm), flush=True)

                # Function that computes the action of the (linearized) true constraints minus the auxiliary constraints
                def v_mat_action(vec):

                    # Build Firedrake functions corresponding to vec and the current guess
                    vec_func = Function(Z_h)
                    with vec_func.dat.vec_wo as vec_func_data:
                        vec.copy(vec_func_data)

                    current_func = Function(Z_h)
                    with current_func.dat.vec_wo as current_func_data:
                        snes.getSolution().copy(current_func_data)

                    # Build pressure and mole fraction functions corresponding to vec and the current guess
                    X_h = P_h * X_h_1 * X_h_2
                    
                    vec_func_p_mfs = Function(X_h)
                    vec_func_p, vec_func_x_1, vec_func_x_2 = vec_func_p_mfs.subfunctions
                    vec_func_p.assign(vec_func.subfunctions[5])
                    vec_func_x_1.assign(vec_func.subfunctions[6])
                    vec_func_x_2.assign(vec_func.subfunctions[7])
                    vec_func_p, vec_func_x_1, vec_func_x_2 = split(vec_func_p_mfs)

                    current_p_mfs = Function(X_h)
                    current_p, current_x_1, current_x_2 = current_p_mfs.subfunctions
                    current_p.assign(current_func.subfunctions[5])
                    current_x_1.assign(current_func.subfunctions[6])
                    current_x_2.assign(current_func.subfunctions[7])
                    current_p, current_x_1, current_x_2 = split(current_p_mfs)

                    # The action of evaluating the chemical potentials and pressure at the auxillary point
                    vec_func_mu_1 = vec_func.subfunctions[3]
                    vec_func_mu_2 = vec_func.subfunctions[4]
                    vec_func_p = vec_func.subfunctions[5]

                    vec_func_mu_1_bc_val = auxiliary_point_bcs[0].eval_at_point(vec_func_mu_1)
                    vec_func_mu_2_bc_val = auxiliary_point_bcs[1].eval_at_point(vec_func_mu_2)
                    vec_func_p_bc_val = auxiliary_point_bcs[2].eval_at_point(vec_func_p)

                    # The (linearization of the) action of evaluating the true constraints
                    current_c_tot, current_c_1, current_c_2 = conc_relation(current_x_1, current_x_2, current_p)

                    c_1_integral_form = current_c_1 * dx
                    c_2_integral_form = current_c_2 * dx
                    mfs_integral_form = (current_x_1 + current_x_2)* dx

                    c_1_integral_lin = derivative(c_1_integral_form, current_p_mfs, vec_func_p_mfs)
                    c_2_integral_lin = derivative(c_2_integral_form, current_p_mfs, vec_func_p_mfs)
                    mfs_integral_lin = derivative(mfs_integral_form, current_p_mfs, vec_func_p_mfs)

                    return np.array([float(assemble(c_1_integral_lin)) - vec_func_mu_1_bc_val,
                                    float(assemble(c_2_integral_lin)) - vec_func_mu_2_bc_val,
                                    float(assemble(mfs_integral_lin)) - vec_func_p_bc_val])

                # Form and invert the 3x3 matrix used in applying Woodbury's identity
                v_u0 = v_mat_action(u0_vec)
                vw_mat = np.transpose(np.array([v_mat_action(w1_vec), v_mat_action(w2_vec), v_mat_action(w3_vec)]))

                I_vw_mat = np.identity(3) + vw_mat
                alpha = npla.solve(I_vw_mat, v_u0)

                cond = npla.cond(I_vw_mat)
                PETSc.Sys.Print("Condition number of the Woodbury matrix is %.5e" % cond, flush=True)

                # Compute the true post-Woodbury solution
                new_sln_vec = snes.getSolution()
                new_sln_vec.axpy(alpha[0], w1_vec)
                new_sln_vec.axpy(alpha[1], w2_vec)
                new_sln_vec.axpy(alpha[2], w3_vec)

                u0_vec.axpy((-1.0) * alpha[0], w1_vec)
                u0_vec.axpy((-1.0) * alpha[1], w2_vec)
                u0_vec.axpy((-1.0) * alpha[2], w3_vec)

                # Compute the true post-Woodbury residual norm (it differs from the PETSc SNES residual norm)
                f_vec = Z_h.dof_dset.layout_vec.copy()

                global actually_evaluate
                actually_evaluate = True
                snes.computeFunction(new_sln_vec, f_vec)

                f_norm = f_vec.norm(norm_type=petsc4py.PETSc.NormType.N2)
                PETSc.Sys.Print("True post-Woodbury residual norm: %.5e" % f_norm, flush=True)

                # Decide if nonlinear convergence has been achieved
                if f_norm <= newton_atol:
                    PETSc.Sys.Print("Newton solver has converged: sufficiently small residual norm", flush=True)
                    return petsc4py.PETSc.SNES.ConvergedReason.CONVERGED_FNORM_ABS
                elif snes.getIterationNumber() > newton_max_it:
                    return petsc4py.PETSc.SNES.ConvergedReason.DIVERGED_MAX_IT
                else:
                    return None

            newton_solver.snes.setConvergenceTest(convergence_test_woodbury)

            # ----------------------- Do the Newton solve ---------------------

            # The main Newton solve
            newton_solver.solve()
            PETSc.Sys.Print("Done Newton system (Refinement %d, N_mesh = %d) \n" % (i, N_mesh), flush=True)

            # Check that the true constraints were enforced
            mm_1, mm_2, v, mu_1, mu_2, p, x_1, x_2, rho_inv = sln.subfunctions
            c_tot, c_1, c_2 = conc_relation(x_1, x_2, p)

            assert(float(abs(assemble(c_1 * dx) - c_1_integral)) < 1e-7)
            assert(float(abs(assemble(c_2 * dx) - c_2_integral)) < 1e-7)
            assert(float(abs(assemble((x_1 + x_2) * dx) - mfs_integral)) < 1e-7)

        # ------------------------ Build the initial guess --------------------

        sln = Function(Z_h)

        mm_1, mm_2, v, mu_1, mu_2, p, x_1, x_2, rho_inv = sln.subfunctions

        # The initial guess is the L^2 projection of the exact solution
        mm_1.assign(project_util(mm_1_ms, W_h_1))
        mm_2.assign(project_util(mm_2_ms, W_h_2))
        v.assign(project_util(v_ms, V_h))
        mu_1.assign(project_util(mu_1_ms, U_h_1))
        mu_2.assign(project_util(mu_2_ms, U_h_2))
        p.assign(project_util(p_ms, P_h))
        x_1.assign(project_util(x_1_ms, X_h_1))
        x_2.assign(project_util(x_2_ms, X_h_2))
        rho_inv.assign(project_util(rho_inv_ms, R_h))

        # -------------------- Do the Newton solve and return -----------------

        newton_solve(sln)
        return sln.subfunctions

    # ------------------ Solve the Picard linearized problem ------------------

    def solve_linear():

        # The mixed function space for the Picard linearized problem
        Z_h = W_h_1 * W_h_2 * V_h * U_h_1 * U_h_2 * P_h

        # Modify T_1 and T_2 to exactly satisfy the compatibility condition at the discrete level
        cc_aux_J_1 = project_util(g_1, W_h_1, bcs=[DirichletBC(W_h_1, g_1, bc_markers)])
        cc_aux_J_2 = project_util(g_2, W_h_2, bcs=[DirichletBC(W_h_2, g_2, bc_markers)])

        Sm_h = FunctionSpace(mesh, "DG" if mesh_type == "tet" else "DQ", k + 2)
        r_1_discrete = project_util(r_1, Sm_h)
        r_2_discrete = project_util(r_2, Sm_h)

        T_1.assign(M_1 * assemble(r_1_discrete * dx) / assemble(inner(cc_aux_J_1, nml) * ds))
        T_2.assign(M_2 * assemble(r_2_discrete * dx) / assemble(inner(cc_aux_J_2, nml) * ds))

        PETSc.Sys.Print(f"Compatibility condition on fluxes - T_1 is: {float(T_1)}", flush=True)
        PETSc.Sys.Print(f"Compatibility condition on fluxes - T_2 is: {float(T_2)}", flush=True)

        # ------------------------- The discrete forms ------------------------

        # Trial and test functions
        mm_1, mm_2, v, mu_1, mu_2, p = TrialFunctions(Z_h)
        u_1, u_2, u, w_1, w_2, q = TestFunctions(Z_h)

        # The Stokes viscous terms
        A_visc = 2.0 * eta * inner(sym(grad(v)), sym(grad(u))) * dx
        A_visc += lame * inner(div(v), div(u)) * dx

        # The augmented Onsager transport matrix terms
        A_osm = (RT / ((c_1_ms + c_2_ms) * D_12)) * ((c_2_ms / (M_1 * M_1 * c_1_ms)) * inner(mm_1, u_1) \
                                        + (c_1_ms / (M_2 * M_2 * c_2_ms)) * inner(mm_2, u_2) \
                                        - (1.0 / (M_1 * M_2)) * (inner(mm_1, u_2) + inner(mm_2, u_1))) * dx
        A_osm += gamma * inner(v - (rho_inv_ms * (mm_1 + mm_2)), u - (rho_inv_ms * (u_1 + u_2))) * dx

        # The diffusion driving force terms and Stokes pressure term
        B_blf = (inner(p, div(rho_inv_ms * (u_1 + u_2))) - inner(p, div(u))) * dx
        B_blf -= ((1.0 / M_1) * inner(mu_1, div(u_1)) + (1.0 / M_2) * inner(mu_2, div(u_2))) * dx

        # The div(mass-average constraint) and continuity equation terms
        BT_blf = (inner(q, div(rho_inv_ms * (mm_1 + mm_2))) - inner(q, div(v))) * dx
        BT_blf -= ((1.0 / M_1) * inner(w_1, div(mm_1)) + (1.0 / M_2) * inner(w_2, div(mm_2))) * dx

        # The total bilinear form
        tot_blf = A_visc + A_osm + B_blf + BT_blf

        # The total linear form
        tot_lf = (inner(f * rho_ms, u) - inner(w_1, r_1_discrete) - inner(w_2, r_2_discrete)) * dx

        # ------------------------- Specify the BCs ---------------------------

        # The BCs on the mass fluxes and barycentric velocity
        flux_bcs = [DirichletBC(Z_h.sub(0), g_1, bc_markers),
                    DirichletBC(Z_h.sub(1), g_2, bc_markers),
                    DirichletBC(Z_h.sub(2), g_v, bc_markers)]

        # Auxiliary point BCs to remove the pressure and chemical potential nullspaces
        auxiliary_point_bcs = [FixAtPointBC(Z_h.sub(3), Constant(0.0), None, bc_point_ref),
                               FixAtPointBC(Z_h.sub(4), Constant(0.0), None, bc_point_ref),
                               FixAtPointBC(Z_h.sub(5), Constant(0.0), None, bc_point_ref)]

        # -------------------- Configure linear solver ------------------------

        linear_solver_parameters = {"ksp_type" : "gmres",
                                    "ksp_max_it" : 3,
                                    "ksp_convergence_test" : "skip",
                                    "pc_type" : "lu",
                                    "pc_factor_mat_solver_type" : "mumps",
                                    "mat_mumps_icntl_14" : 120,
                                    "mat_mumps_icntl_24" : 0,   # No null pivot detection
                                    "ksp_monitor" : ""}
        
        # ---------------------- Do the linear solve --------------------------

        PETSc.Sys.Print("Solve Picard linearized system (Refinement %d, N_mesh = %d)" % (i, N_mesh), flush=True)

        A_sys = assemble(tot_blf, bcs=(flux_bcs + auxiliary_point_bcs))
        b_sys = assemble(tot_lf, bcs=(flux_bcs + auxiliary_point_bcs))

        sln = Function(Z_h)

        solve(A_sys, sln, b_sys, solver_parameters=linear_solver_parameters)

        PETSc.Sys.Print("Done Picard linearized system (Refinement %d, N_mesh = %d) \n" % (i, N_mesh), flush=True)

        # Check that the point BCs were enforced
        mm_1, mm_2, v, mu_1, mu_2, p = sln.subfunctions

        auxiliary_point_bcs[0].assert_is_enforced(mu_1)
        auxiliary_point_bcs[1].assert_is_enforced(mu_2)
        auxiliary_point_bcs[2].assert_is_enforced(p)

        # Shift the chemical potentials and pressure to have same mean as exact solution
        mu_1 += Constant(assemble((mu_1_ms - mu_1) * dx) / vol)
        mu_2 += Constant(assemble((mu_2_ms - mu_2) * dx) / vol)
        p += Constant(assemble((p_ms - p) * dx) / vol)

        # Return the solution
        return sln.subfunctions

    # ----------------------------- Compute the errors ------------------------

    if picard_linearized:
        mm_1, mm_2, v, mu_1, mu_2, p = solve_linear()

    else:
        mm_1, mm_2, v, mu_1, mu_2, p, x_1, x_2, rho_inv = solve_nonlinear()

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

    if picard_linearized:   # x_1, x_2 and rho_inv are not unknowns in the Picard linearized problem
        mass_avg_err = l2_dist(v, rho_inv_ms * (mm_1 + mm_2))

        x_1_err = np.nan
        x_2_err = np.nan
        rho_inv_err = np.nan

    else:
        mass_avg_err = l2_dist(v, rho_inv * (mm_1 + mm_2))

        x_1_err = l2_dist(x_1_ms, x_1)
        x_2_err = l2_dist(x_2_ms, x_2)

        rho_inv_err = l2_dist(rho_inv_ms, rho_inv)

    # Store the absolute and relative errors
    errs[i, :, 0] = np.array([mu_1_err, mu_2_err, grad_mu_1_err, grad_mu_2_err,
                              p_err, grad_p_err,
                              mm_1_err, mm_2_err, div_mm_1_err, div_mm_2_err,
                              v_err, grad_v_err, mass_avg_err,
                              x_1_err, x_2_err, rho_inv_err])

    norms = np.array([l2_norm(mu_1_ms), l2_norm(mu_2_ms), l2_norm(grad(mu_1_ms)), l2_norm(grad(mu_2_ms)),
                      l2_norm(p_ms), l2_norm(grad(p_ms)),
                      l2_norm(mm_1_ms), l2_norm(mm_2_ms), l2_norm(div(mm_1_ms)), l2_norm(div(mm_2_ms)),
                      l2_norm(v_ms), l2_norm(grad(v_ms)), l2_norm(v_ms),
                      l2_norm(x_1_ms), l2_norm(x_2_ms), l2_norm(rho_inv_ms)])

    errs[i, :, 1] = np.multiply(errs[i, :, 0], 1.0 / norms)

# ---------------------------- Display the errors and rates -------------------

# Save the errors
import pathlib
pathlib.Path("manufactured_solution_out").mkdir(parents=True, exist_ok=True)
np.save("manufactured_solution_out/errs.npy", errs, allow_pickle=False)

# The name of the fields
errs_names = ["mu_1", "mu_2", "grad_mu_1", "grad_mu_2",
               "p", "grad_p",
               "mm_1", "mm_2", "div_mm_1", "div_mm_2",
               "v", "grad_v", "mass_avg",
               "x_1", "x_2", "rho_inv"]

# The associated theoretical convergence rates (for the linearized problem)
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