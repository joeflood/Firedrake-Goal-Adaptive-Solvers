from firedrake import *
from netgen.occ import *
import csv

initial_mesh_size = 0.2
# Initial mesh
box1 = WorkPlane().MoveTo(-1, 0).Rectangle(1, 1).Face()
box2 = WorkPlane().MoveTo(0, 0).Rectangle(1, 1).Face()
box3 = WorkPlane().MoveTo(0, -1).Rectangle(1, 1).Face()

# Now they are geometric shapes you can combine
shape = box1 + box2 + box3

for f in shape.edges: # Assign face labels
    if f.center.x == -1:
        f.name = "goal_face"
    if f.center.x == 1 or f.center.y == 1:
        f.name = "dirichletbcs"

geo = OCCGeometry(shape, dim = 2)
ngmesh = geo.GenerateMesh(maxh=initial_mesh_size)
mesh = Mesh(ngmesh)
VTKFile("adaptivemesh_unrefined.pvd").write(mesh)

# Solver parameters
degree = 1
max_iterations = 30
dual_solve_method = "high_order" # Options: high_order, star
residual_solve_method = "automatic" # Options: manual, automatic
residual_degree = degree + 0 # Degree of residuals 
dorfler_alpha = 0.5
tolerance = 0.0001

# Output vectors
N_vec = []
eta_vec = []
etah_vec = []
etaTsum_vec = []
eff1_vec = []
eff2_vec = []

def boundary_labels(mesh):
    ngmesh = mesh.netgen_mesh
    names = ngmesh.GetRegionNames(codim=1)
    names_to_labels = {}
    for l in names:
        names_to_labels[l] = tuple(i+1 for i, name in enumerate(names) if name == l)
    return names_to_labels

def residual(F, test): # Residual helper function
    v = F.arguments()[0]
    return replace(F, {v: test})

def build_primal_problem(mesh): # Define PDE problem & Goal Functional
    V = FunctionSpace(mesh, "CG", degree, variant="integral") # Template function space used to define the PDE
    u = Function(V, name="Solution")
    v = TestFunction(V)
    coords = SpatialCoordinate(u.function_space().mesh()) # MMS Method of Manufactured Solution
    x, y = coords[0], coords[1]
    u_exact = (x-1)*(y-1)**2
    G = as_vector(((y-1)**2, 2*(x-1)*(y-1)))
    n = FacetNormal(mesh)
    g = dot(G,n)
    f = -div(grad(u_exact))

    labels = boundary_labels(mesh)
    ds_goal = Measure("ds", domain=mesh, subdomain_id=labels['goal_face'])

    F = inner(grad(u), grad(v))*dx - inner(f, v)*dx - g*v*ds
    bcs = [DirichletBC(V, u_exact, labels['dirichletbcs'])]
    
    J = dot(grad(u), n)*ds_goal
    data = dict(V=V, u=u, v=v, u_exact=u_exact, f=f, n=n, J_form=J, bcs=bcs, F=F)
    return data

# From here we repeat
for it in range(max_iterations):    
    print(f"Solving on level {it}")
    VTKFile(f"output/mesh{it}.pvd").write(mesh)
    
    prob = build_primal_problem(mesh)
    V      = prob["V"]
    u      = prob["u"]
    f      = prob["f"]
    v      = prob["v"]
    u_exact= prob["u_exact"]
    F      = prob["F"]
    bcs    = prob["bcs"]
    J = prob["J_form"]

    # Obtain degrees of freedom
    ndofs = V.dim()
    print("N:" , ndofs)
    N_vec.append(ndofs)
    
    # 2. Compute the finite element solution of the primal problem on the current mesh.
    print("Solving primal ...")
    solve(F == 0, u, bcs=bcs)
    print("Writing primal solution ...")
    VTKFile(f"output/solution_{it}.pvd").write(u)
    
    print("Solving dual ...")
    # 3. & 4. Solve the dual problem
    # Solver parameters
    sp_chol = {"pc_type": "cholesky",
            "pc_factor_mat_solver_type": "mumps"}
    sp_pmg = {"snes_type": "ksponly",
            "ksp_type": "cg",
            "ksp_rtol": 1.0e-10,
            "ksp_max_it": 1,
            "ksp_convergence_test": "skip",
            "ksp_monitor": None,
            "pc_type": "python",
            "pc_python_type": "firedrake.P1PC",
            "pmg_mg_coarse": {
                "pc_type": "python",
                "pc_python_type": "firedrake.AssembledPC",
                "assembled_pc_type": "cholesky",
            },
            "pmg_mg_levels": {
                "ksp_max_it": 1,
                "ksp_type": "chebyshev",
                "pc_type": "python",
                "pc_python_type": "firedrake.ASMStarPC",
                "pc_star_mat_ordering_type": "metisnd",
                "pc_star_sub_sub_pc_type": "cholesky",
            }
        }
    sp_star = {"snes_type": "ksponly",
            "ksp_type": "cg",
            "ksp_rtol": 1.0e-10,
            "ksp_max_it": 5,
            "ksp_convergence_test": "skip",
            "ksp_monitor": None,
            "pc_type": "python",
            "pc_python_type": "firedrake.ASMStarPC",
            "pc_star_mat_ordering_type": "metisnd",
            "pc_star_sub_sub_pc_type": "cholesky",
            }

    # Solve dual in degree + 1
    Vf = FunctionSpace(mesh, "Lagrange", degree + 1) #Dual function space
    vz = TestFunction(Vf) # Dual test function
    z = Function(Vf) # Dual soluton

    G = action(adjoint(derivative(F, u, TrialFunction(Vf))), z) - derivative(J, u, vz)
    G = replace(G, {v: vz})
    bcs_dual  = [bc.reconstruct(V=Vf, g=0) for bc in bcs]
    
    if dual_solve_method == "high_order":
        solve(G == 0, z, bcs_dual, solver_parameters=sp_chol) # Obtain z
        z_lo = Function(V, name="LowOrderDualSolution")
        z_lo.interpolate(z)
        z_err = z - z_lo
    
    elif dual_solve_method == "star":
        solve(G == 0, z, bcs_dual, solver_parameters=sp_star)
        z_err = z    
    
    else:
        print("ERROR: Unknown dual solve method.")
        break

    # 5. & 6. Compute eta_h to determine whether to continue.
    Juh = assemble(J)
    Ju = assemble(replace(J, {u: u_exact}))
    eta_h = abs(assemble(residual(F, z)))
    eta = abs(Juh - Ju)
    etah_vec.append(eta_h)
    eta_vec.append(eta)
    print(f"J(u): {Ju}, J(uh) = {Juh}")
    print(f"eta = {eta}")
    print(f"eta_h = {eta_h}")

    if np.abs(eta_h) < tolerance:
        print("Error estimate below tolerance, finished.")
        break
    
    # 7. Compute cell and facet residuals R_T, R_\partialT
    if residual_solve_method == "automatic":
        cell = mesh.ufl_cell()  #Returns the cell from the mesh
        dim = mesh.topological_dimension() # Dimension of the mesh 
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
        print("Computing Rcells ...")
        solve(ac == Lc, Rcell) # solve for Rcell polynonmial

        def both(u):
            return u("+") + u("-")

        # ---------------- Equation 4.8 to find facet residual Rfacet -------------------------
        FB = FunctionSpace(mesh, "FB", dim, variant=variant) # Cone function space
        cones = Function(FB).assign(1) # Cones

        el = BrokenElement(FiniteElement("FB", cell=cell, degree=residual_degree+dim, variant=variant))
        Q = FunctionSpace(mesh, el)
        q = TestFunction(Q)
        p = TrialFunction(Q)
        Lf = residual(F, q) - inner(Rcell, q)*dx
        af = both(inner(p/cones, q))*dS + inner(p/cones, q)*ds

        Rhat = Function(Q)
        print("Computing Rhats ...")
        solve(af == Lf, Rhat)
        Rfacet = Rhat/cones

        # 8. Compute error indicators eta_T 
        DG0 = FunctionSpace(mesh, "DG", degree=0)
        test = TestFunction(DG0)
        vol = CellVolume(mesh)
        etaT = Function(DG0)

        #eta_T = assemble((inner(test*Rcell, z_err)*dx +  avg(inner(test*Rfacet,z_err))*dS + inner(test*Rfacet,z_err)*ds))

        G = (
            inner(etaT / vol, test)*dx
            - inner(inner(Rcell, z_err), test)*dx + 
            - inner(avg(inner(Rfacet,z_err)), both(test))*dS + 
            - inner(inner(Rfacet,z_err), test)*ds
         )

        print("Computing eta_T indicators ...")
        sp = {"mat_type": "matfree", "ksp_type": "richardson", "pc_type": "jacobi"}
        solve(G == 0, etaT, solver_parameters=sp)

    elif residual_solve_method == "manual":
        DG0 = FunctionSpace(mesh, "DG", degree=0)
        test = TestFunction(DG0)
        vol = CellVolume(mesh)
        etaT_manual = Function(DG0)
        n = FacetNormal(mesh)

        H = (
            inner(etaT / vol, test)*dx
            - inner(f + div(grad(u)), z_err * test) * dx
            - 0.5 * inner(jump(-grad(u), n), z_err * test('+')) * dS
            - 0.5 * inner(jump(-grad(u), n), z_err * test('-')) * dS
            - inner(dot(-grad(u), n), z_err * test) * ds
        )

        # Each cell is an independent 1x1 solve, so Jacobi is exact
        sp = {"mat_type": "matfree", "ksp_type": "richardson", "pc_type": "jacobi"}
        solve(H == 0, etaT_manual, solver_parameters=sp)

    else: 
        print("ERROR: Unkown residual solve method.")
        break


    # Compute sum etaT
    with etaT.dat.vec as evec:
        evec.abs()    
        etaT_array = evec.getArray()
    
    etaT_total = abs(np.sum(etaT_array))
    etaTsum_vec.append(etaT_total)
    print(f"sum_T(eta_T): {etaT_total}")

    # Compute efficiency indices
    eff1 = eta_h/eta
    eff2 = etaT_total/eta
    print(f"Efficiency index 1 = {eff1}")
    print(f"Efficiency index 2 = {eff2}")
    eff1_vec.append(eff1)
    eff2_vec.append(eff2)

    # 9. Mark cells for refinement (Dorfler marking)
    print("Marking cells for refinement ...")
    sorted_indices = np.argsort(-etaT_array)
    sorted_etaT = etaT_array[sorted_indices]
    cumulative_sum = np.cumsum(sorted_etaT)
    threshold = dorfler_alpha * etaT_total
    M = np.searchsorted(cumulative_sum, threshold) + 1
    marked_cells = sorted_indices[:M]

    markers_space = FunctionSpace(mesh, "DG", 0)
    markers = Function(markers_space)
    with markers.dat.vec as mv:
        marr = mv.getArray()
        marr[:] = 0
        marr[marked_cells] = 1
    
    # 10. Refine cells marked for refinement
    if it == max_iterations - 1:
        print(f"Maximum iteration ({max_iterations}) reached. Exiting.")
        break

    print("Remeshing ...")
    mesh = mesh.refine_marked_elements(markers)

rows = list(zip(N_vec, eta_vec, etah_vec, etaTsum_vec, eff1_vec, eff2_vec))
headers = ("N", "eta", "eta_h", "sum_eta_T", "eff1", "eff2")
with open("results.csv", "w", newline="") as file:
    w = csv.writer(file)
    w.writerow(headers)
    w.writerows(rows)   