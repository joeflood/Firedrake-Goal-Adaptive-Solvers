from firedrake import *
from netgen.occ import *

def solve_dual(mesh, F, bc, M, method = "deg"):
    if method == "deg":
        W = FunctionSpace(mesh, "Lagrange", 2) #Dual function space
        phi = TrialFunction(W)
        z = Function(W) # Dual soluton
        w = TestFunction(W) # Dual test function

        M = replace(M, {M.argument(), w})
        
        a_adj = adjoint(derivative(F, phi))
        bc.homogenize()

        solve(a_adj == M, z, bc)
    elif method == "approx":

        print("Approximate method not implemented yet.")
    else:
        print("Unkown dual solve method.")
    
    return z

def estimate_goal_error(mesh, uh, F, M): #This is where the automatic part comes in
    f = Constant(1)
    z = solve_dual(mesh, F, M)
    
    v = CellVolume(mesh)
    DG0 = FunctionSpace(mesh, "DG", 0)
    eta_signed = Function(DG0)
    w = TestFunction(DG0)

    g = inner(f,z) - inner(grad(uh), grad(z))
    G = inner(eta_signed / v, w) * dx - inner(g, w) * dx
    
    sp = {"mat_type": "matfree", "ksp_type": "richardson", "pc_type": "jacobi"}
    solve(G == 0, eta_signed, solver_parameters=sp)

    eta = Function(DG0, name="goal_indicators")
    eta.interpolate(abs(eta_signed))

    with eta.dat.vec_ro as eta_: # compute estimate for error in energy norm
        error_est = sqrt(eta_.dot(eta_))
    return (eta, error_est)

def adapt(mesh, eta):
    W = FunctionSpace(mesh, "DG", 0)
    markers = Function(W)

    with eta.dat.vec_ro as eta_:
        eta_max = eta_.max()[1]

    theta = 0.5
    should_refine = conditional(gt(eta, theta*eta_max), 1, 0)
    markers.interpolate(should_refine)

    refined_mesh = mesh.refine_marked_elements(markers)
    return refined_mesh

def adaptive_solve(mesh, F, uh, bc, M, tol = 10^-3, max_it = 10):
    error_estimators = []
    dofs = []

    for i in range(max_iterations):
        print(f"Solving on level {i}")

        solve(F == 0, uh, bc)
        VTKFile(f"output/adaptive_loop_{i}.pvd").write(uh)
        (eta, error_est) = estimate_goal_error(mesh, uh, F, M)
        print(f" ||u - u_h||: {error_est}")

        if error_est < tol:
            print(f"Estimated error in simulation is < {tol}, simulation stopped.")
            break

        error_estimators.append(error_est)
        dofs.append(uh.function_space().dim())
        mesh = adapt(mesh, eta)
        VTKFile(f"output/adaptive_mesh_{i}.pvd").write(mesh)



# Make 2D rectangle from (0, 0) to (1, 2)
rect1 = WorkPlane(Axes((0,0,0), n=Z, h=X)).Rectangle(1,2).Face()
# Make 2D rectangle from (0, 1) to (2, 2)
rect2 = WorkPlane(Axes((0,1,0), n=Z, h=X)).Rectangle(2,1).Face()
L = rect1 + rect2
geo = OCCGeometry(L, dim=2)
ngmesh = geo.GenerateMesh(maxh=0.1)
mesh = Mesh(ngmesh)
File("adaptivemesh/unrefined.pvd").write(mesh)

V = FunctionSpace(mesh, "Lagrange", 1)
u = Function(V, name = "solution")
v = TestFunction(V)
n = FacetNormal(mesh)
ds = Measure("ds", domain=mesh) 

f = 1
bc = DirichletBC(V, 0, "on_boundary")
F = inner(grad(u), grad(v))*dx - f*v*dx
M = inner(grad(v), n) * ds

max_iterations = 10
tol = 0.1

dual_sol = solve_dual(mesh, F, bc, M)

VTKFile()
