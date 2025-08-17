# This script solves the 2D linearized SOSM problem (i.e. the concentrations are given)
# For simplicity only consider the two species case

import sys
import shutil
from firedrake import *
from firedrake.petsc import PETSc
import firedrake.utils as firedrake_utils
import numpy as np
#from tabulate import tabulate

# -------------- Some key parameters relating to the problem setup -------------

d = 2                       # The spatial dimension
assert d == 2

k = 4                       # The polynomial degree (for the velocity spaces)
deg_max = 8 * k             # Maximum quadrature degree

mesh_type = "triangle"      # What mesh type to use
assert mesh_type in ["triangle", "quad"]

use_discrete_conc_and_rho = True    # Use a discrete projection of the concentrations and inverse density?

# -------------- Looping over meshes to get convergence rates-------------------

n_loops = 5

num_err_fields = 13
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
    def l2_norm(f):
        return np.sqrt(assemble(inner(f, f) * dx(mesh, degree=deg_max)))

    def l2_dist(f, g):
        return l2_norm(f - g)

    # Utility class for fixing a (possibly DG) field at a node
    class FixAtPointBC(DirichletBC):
        def __init__(self, V, g, sub_domain, bc_point):
            super().__init__(V, g, sub_domain)
            self.bc_point = bc_point
            self.min_index = None

        @firedrake_utils.cached_property
        def nodes(self):
            V = self.function_space()
            x_sc = SpatialCoordinate(V.mesh())
            point_dist = interpolate(sqrt(dot(x_sc - self.bc_point, x_sc - self.bc_point)), V)

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
            bc_value = self.function_arg(self.bc_point)
            V_func_value = self.eval_at_point(V_func)

            assert abs(bc_value - V_func_value) < 1e-8

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

    zeta = Constant(1e-1)                   # Bulk viscosity for Stokes
    eta = Constant(1e-1)                    # Shear viscosity for Stokes
    lame = (zeta - ((2.0 * eta) / d))       # Lame parameter

    # Other parameters
    gamma = Constant(1e1)       # The augmentation parameter

    # ----------------------------- The manufactured solution ------------------
    x = SpatialCoordinate(mesh)

    nml = FacetNormal(mesh)
    Id = Identity(d)

    # TODO: Do we get strange convergence rates by changing g_ms_const, like for the stress discretization?
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

    epsilon_v_ms = sym(grad(v_ms))                                      # Strain rate
    tau_ms = (2.0 * eta * epsilon_v_ms) + \
              lame * tr(epsilon_v_ms) * Id                              # Viscous stress
    p_ms = c_T_ms                                                       # Pressure

    f_ms = (grad(p_ms) - div(tau_ms))                                   # Stokes forcing term (times the density)

    # ----------------------------- The problem data ---------------------------
    f = f_ms / rho_ms                           # Stokes forcing term

    r_1 = div(c_1_ms * v_1_ms)                  # Volumetric generation/depletion rate of species 1
    r_2 = div(c_2_ms * v_2_ms)                  # Volumetric generation/depletion rate of species 2

    g_v = v_ms                                  # Boundary condition for the bulk velocity

    g_1 = mm_1_ms                               # (Normal) boundary condition for species 1 momentum
    g_2 = mm_2_ms                               # (Normal) boundary condition for species 2 momentum

    # The boundary point at which the "reference" pressure and chemical potentials are measured
    bc_point_ref = Constant([0.0, 0.0])

    mu_1_ref = Constant(mu_1_ms(bc_point_ref))
    mu_2_ref = Constant(mu_2_ms(bc_point_ref))
    p_ref = Constant(p_ms(bc_point_ref))

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
        W_h_1 = FunctionSpace(mesh, "RT", k)            # Momentum space for species 1
        W_h_2 = FunctionSpace(mesh, "RT", k)            # Momentum space for species 2
        V_h   = VectorFunctionSpace(mesh, "CG", k)      # Bulk velocity space
        X_h_1 = FunctionSpace(mesh, "DG", k - 1)        # Chemical potential space for species 1
        X_h_2 = FunctionSpace(mesh, "DG", k - 1)        # Chemical potential space for species 2
        P_h   = FunctionSpace(mesh, "CG", k - 1)        # Pressure space

    elif mesh_type == "quad":
        var = "equispaced"
        W_h_1 = FunctionSpace(mesh, FiniteElement("RTCF", quadrilateral, k, variant=var))       # Momentum space for species 1
        W_h_2 = FunctionSpace(mesh, FiniteElement("RTCF", quadrilateral, k, variant=var))       # Momentum space for species 2
        V_h   = VectorFunctionSpace(mesh, "CG", k)                                              # Bulk velocity space
        X_h_1 = FunctionSpace(mesh, FiniteElement("DQ", quadrilateral, k - 1, variant=var))     # Chemical potential space for species 1
        X_h_2 = FunctionSpace(mesh, FiniteElement("DQ", quadrilateral, k - 1, variant=var))     # Chemical potential space for species 2
        P_h   = FunctionSpace(mesh, "CG", k - 1)                                                # Pressure space

    # Discrete product space
    Z_h = W_h_1 * W_h_2 * V_h * X_h_1 * X_h_2 * P_h

    # ---------------- The discrete concentrations and inverse density ---------

    if use_discrete_conc_and_rho:

        C_h = FunctionSpace(mesh, "DG" if mesh_type == "triangle" else "DQ", k - 1)
        R_h = FunctionSpace(mesh, "CG", k - 1)

        c_1_h = project_util(c_1_ms, C_h)
        c_2_h = project_util(c_2_ms, C_h)
        rho_inv_h = project_util(1 / (M_1 * c_1_h + M_2 * c_2_h), R_h)

    else:
        c_1_h = c_1_ms
        c_2_h = c_2_ms
        rho_inv_h = 1 / rho_ms

    # ---- Modify source terms to preserve chemical potential nullspace --------
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

    # --------------------------- The discrete forms ---------------------------

    # Trial and test functions
    mm_1, mm_2, v, mu_1, mu_2, p = TrialFunctions(Z_h)
    u_1, u_2, u, w_1, w_2, q = TestFunctions(Z_h)

    # The viscous terms
    A_visc = 2.0 * eta * inner(sym(grad(v)), sym(grad(u))) * dx(degree=deg_max)
    A_visc += lame * inner(div(v), div(u)) * dx(degree=deg_max)

    # The OSM terms
    A_osm = (RT / ((c_1_h + c_2_h) * D_12)) * ((c_2_h / (M_1 * M_1 * c_1_h)) * inner(mm_1, u_1) \
                                      + (c_1_h / (M_2 * M_2 * c_2_h)) * inner(mm_2, u_2) \
                                      - (1.0 / (M_1 * M_2)) * (inner(mm_1, u_2) + inner(mm_2, u_1))) * dx(degree=deg_max)
    A_osm += gamma * inner(v - (rho_inv_h * (mm_1 + mm_2)), u - (rho_inv_h * (u_1 + u_2))) * dx(degree=deg_max)

    # The diffusion driving force terms (and their transposes)
    B_blf = (inner(p, div(rho_inv_h * (u_1 + u_2))) - inner(p, div(u))) * dx(degree=deg_max)
    B_blf -= ((1.0 / M_1) * inner(mu_1, div(u_1)) + (1.0 / M_2) * inner(mu_2, div(u_2))) * dx(degree=deg_max)

    BT_blf = (inner(q, div(rho_inv_h * (mm_1 + mm_2))) - inner(q, div(v))) * dx(degree=deg_max)
    BT_blf -= ((1.0 / M_1) * inner(w_1, div(mm_1)) + (1.0 / M_2) * inner(w_2, div(mm_2))) * dx(degree=deg_max)

    # The total bilinear form
    tot_blf = A_visc + A_osm + B_blf + BT_blf

    # The density consistency terms
    tot_blf -= p * inner((rho_inv_h * (u_1 + u_2)) - u, nml) * ds(degree=deg_max)
    tot_blf -= q * inner((rho_inv_h * (mm_1 + mm_2)) - v, nml) * ds(degree=deg_max)

    # The forcing terms
    tot_func = (inner(f * (M_1 * c_1_h + M_2 * c_2_h), u) - inner(w_1, r_1_discrete) - inner(w_2, r_2_discrete)) * dx(degree=deg_max)

    # ----------------------------- Solve the linear system --------------------

    # Specify the BCs
    bcs = [DirichletBC(Z_h.sub(0), g_1, (1, 2, 3, 4)),
           DirichletBC(Z_h.sub(1), g_2, (1, 2, 3, 4)),
           DirichletBC(Z_h.sub(2), g_v, (1, 2, 3, 4)),
           FixAtPointBC(Z_h.sub(3), mu_1_ref, None, bc_point_ref),
           FixAtPointBC(Z_h.sub(4), mu_2_ref, None, bc_point_ref),
           FixAtPointBC(Z_h.sub(5), p_ref, None, bc_point_ref)]
    
    # Assemble the system
    PETSc.Sys.Print("Assemble linear system (Refinement %d, N_mesh = %d)" % (i, N_mesh), flush=True)

    A_sys = assemble(tot_blf, bcs=bcs)
    b_sys = assemble(tot_func, bcs=bcs)

    # Solve the system
    sln = Function(Z_h)                     # The discrete solution vector

    PETSc.Sys.Print("Solve linear system (Refinement %d, N_mesh = %d)" % (i, N_mesh), flush=True)

    solver_parameters = {"ksp_type" : "gmres",
                         "ksp_max_it" : 3,
                         "ksp_convergence_test" : "skip",
                         "pc_type" : "lu",
                         "pc_factor_mat_solver_type" : "mumps",
                         "mat_mumps_icntl_14": 120,
                         "ksp_monitor" : ""}

    solve(A_sys, sln, b_sys, solver_parameters=solver_parameters)

    PETSc.Sys.Print("Done solving linear system (Refinement %d, N_mesh = %d) \n" % (i, N_mesh), flush=True)

    # Check that the point BCs were enforced
    mm_1, mm_2, v, mu_1, mu_2, p = sln.subfunctions
    bcs[-3].assert_is_enforced(mu_1)
    bcs[-2].assert_is_enforced(mu_2)
    bcs[-1].assert_is_enforced(p)

    # ----------------------------- Compute the errors -------------------------
    mm_1, mm_2, v, mu_1, mu_2, p = sln.subfunctions

    # Generate Paraview plots of the errors?
    plot_errors = True
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

        mm_1_err.assign(project_util(mm_1 - mm_1_ms, ErrVSV))
        mm_2_err.assign(project_util(mm_2 - mm_2_ms, ErrVSV))
        v_err.assign(project_util(v - v_ms, ErrVSV))
        mu_1_err.assign(project_util(mu_1 - mu_1_ms, ErrVSS))
        mu_2_err.assign(project_util(mu_2 - mu_2_ms, ErrVSS))
        p_err.assign(project_util(p - p_ms, ErrVSS))
        mass_avg_err.assign(project_util(v - rho_inv_h * (mm_1 + mm_2), ErrVSV))

        if Z_h.mesh().comm.rank == 0 and i == 0:
            try:
                shutil.rmtree("linearized_2d_out")
            except:
                pass

        File("linearized_2d_out/refinement_%d.pvd" % i).write(mm_1_err, mm_2_err, v_err, mu_1_err, mu_2_err, p_err, mass_avg_err)

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

    mass_avg_err = l2_dist(v, rho_inv_h * (mm_1 + mm_2))

    # Store the absolute and relative errors
    errs[i, :, 0] = np.array([mu_1_err, mu_2_err, grad_mu_1_err, grad_mu_2_err,
                              p_err, grad_p_err,
                              mm_1_err, mm_2_err, div_mm_1_err, div_mm_2_err,
                              v_err, grad_v_err, mass_avg_err])

    norms = np.array([l2_norm(mu_1_ms), l2_norm(mu_2_ms), l2_norm(grad(mu_1_ms)), l2_norm(grad(mu_2_ms)),
                      l2_norm(p_ms), l2_norm(grad(p_ms)),
                      l2_norm(mm_1_ms), l2_norm(mm_2_ms), l2_norm(div(mm_1_ms)), l2_norm(div(mm_2_ms)),
                      l2_norm(v_ms), l2_norm(grad(v_ms)), l2_norm(v_ms)])

    errs[i, :, 1] = np.multiply(errs[i, :, 0], 1.0 / norms)



sys.exit()
# ----------------------------- Display the errors and rates -------------------

# Save the errors to numpy format?
save_errors = False
if save_errors:
    np.save("linearized_2d_out/errs.npy", errs, allow_pickle=False)

# The name of the fields
errs_names = ["mu_1", "mu_2", "grad_mu_1", "grad_mu_2",
               "p", "grad_p",
               "mm_1", "mm_2", "div_mm_1", "div_mm_2",
               "v", "grad_v", "mass_avg"]

# The associated theoretical convergence rates
rates_theory = [str(k), str(k), "-", "-",
                str(k), "-",
                str(k), str(k), str(k), str(k),
                str(k), str(k), str(k)]

# The associated optimal convergence rates
rates_optimal = [str(k), str(k), str(k - 1), str(k - 1),
                 str(k), str(k - 1),
                 str(k + 1), str(k + 1), str(k), str(k),
                 str(k + 1), str(k), str(k + 1)]

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
               "v", "\\nabla v", "v - \\sum_i \\omega_i v_i"]

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
