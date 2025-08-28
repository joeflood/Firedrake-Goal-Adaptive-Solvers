from firedrake import *
from netgen.occ import *
import matplotlib
import numpy as np

nx = 100
mesh = Mesh(unit_square.GenerateMesh(maxh=1/nx))
coords = SpatialCoordinate(mesh) 
x, y = coords[0], coords[1]

# Define actual problem ----------------------- degree 1
degree = 1
V = FunctionSpace(mesh, "CG", degree) # Template function space used to define the PDE
u = TrialFunction(V)
v = TestFunction(V)
A = inner(grad(u), grad(v)) * dx
M = inner(u,v) * dx
bcs = [DirichletBC(V, 0.0, "on_boundary")]

# MMS Method of Manufactured Solution
mmax = 6   # maximum m index
nmax = 6   # maximum n index

# build list of all (lambda_exact, m, n)
exact_list = [
    (np.pi**2 * (m**2 + n**2), m, n)
    for m in range(1, mmax+1)
    for n in range(1, nmax+1)
]

# sort by lambda value
exact_list.sort(key=lambda t: t[0])

# extract just the lambda values into a NumPy array
exact_lambdas = np.array([item[0] for item in exact_list])

eigenproblem = LinearEigenproblem(A, M, bcs)

solver_parameters =  {"eps_type": "krylovschur",
    "eps_which": "smallest_real",
    "st_type": "sinvert",
    "eps_target": 0.0,
    "st_pc_factor_shift_type": "NONZERO"}

n_evals = 3

eigensolver = LinearEigensolver(eigenproblem, n_evals=n_evals, solver_parameters=solver_parameters) 
eigensolver.solve()

lam = []
eigenfunction_real = []
eigenfunction_imag = []
for i in range(n_evals):
    lam.append(eigensolver.eigenvalue(i))
    re, im = eigensolver.eigenfunction(i)
    eigenfunction_real.append(re)
    eigenfunction_imag.append(im)
    #print("Eigenvalue #", i+1, ": ", lam[i])

# Dual problem
degree_d = degree + 1
V_d = FunctionSpace(mesh, "CG", degree_d) # Template function space used to define the PDE
u_d = TrialFunction(V_d)
v_d = TestFunction(V_d)
A_d = inner(grad(u_d), grad(v_d)) * dx
M_d = inner(u_d,v_d) * dx
bcs_d = [DirichletBC(V_d, 0.0, "on_boundary")]

eigenproblem_d = LinearEigenproblem(A_d, M_d, bcs_d)
eigensolver_d = LinearEigensolver(eigenproblem_d, n_evals=n_evals, solver_parameters=solver_parameters) 
eigensolver_d.solve()

lam_d = []
eigenfunction_real_d = []
eigenfunction_imag_d = []
for i in range(n_evals):
    lam_d.append(eigensolver_d.eigenvalue(i))
    re, im = eigensolver_d.eigenfunction(i)
    eigenfunction_real_d.append(re)
    eigenfunction_imag_d.append(im)
    #print("Dual eigenvalue #", i+1, ": ", lam_d[i])

#for i in range(n_evals):

    #print(i, ":", "Primal lam: ", lam[i] ," dual lam: ", lam_d[i], "difference: ", abs(lam[i]-lam_d[i]))

# Error estimators
n = FacetNormal(mesh)
DG0 = FunctionSpace(mesh, "DG", degree=0)
test = TestFunction(DG0)

def both(u):
    return u("+") + u("-")

print("Computing eta_T indicators ...")


def residual_form(u, lam, z_err):
    return (
        inner(div(grad(u)), z_err * test) * dx - 
        lam * inner(u,z_err * test) * dx + 
        inner(0.5*jump(-grad(u), n), z_err * both(test)) * dS +
        inner(dot(-grad(u), n), z_err * test) * ds
    )


z_err_real = []
z_err_imag = []
etaT = []
eta_abs = Function(DG0)
for i in range(n_evals):
    z_err_real.append(eigenfunction_real_d[i] - eigenfunction_real[i]) # would normally take off the interpolant , but this is a self-adjoint problem
    z_err_imag.append(eigenfunction_imag_d[i] - eigenfunction_imag[i])
    eta_real = assemble(residual_form(eigenfunction_real[i], Constant(lam[i].real), z_err_real[i]))
    eta_imag = assemble(residual_form(eigenfunction_imag[i], Constant(lam[i].imag), z_err_imag[i]))

    sigma = 1/2 * assemble(inner(z_err_real[i], z_err_real[i]) * dx + inner(z_err_imag[i], z_err_imag[i]) * dx)

    with eta_abs.dat.vec as v, \
     eta_real.dat.vec_ro as vr, \
     eta_imag.dat.vec_ro as vi:
        v.pointwiseMult(vr, vr)
        tmp = v.duplicate()
        tmp.pointwiseMult(vi, vi)
        v.axpy(1.0, tmp)
        v.sqrtabs()

    # etah calc
    ur, ui = eigenfunction_real[i], eigenfunction_imag[i]
    lam_p = lam[i].real
    zr, zi = eigenfunction_real_d[i], eigenfunction_imag_d[i]
    # Complex residual pairing R_h(z) = a(u_h,z) - lam_h m(u_h,z)
    #eta_h = abs((assemble(inner(grad(ur), grad(z_err_real[i])) * dx) - lam_p *assemble(inner(ur, z_err_real[i]) * dx))/(assemble(inner(ur,z_err_real[i]) * dx)) )
    eta_h = abs( (assemble(inner(grad(ur),grad(z_err_real[i])) * dx) - lam_p*assemble(inner(ur,z_err_real[i]) * dx)) / assemble(inner(z_err_real[i],z_err_real[i]) * dx))
    eta_h_predict = abs(assemble(inner(grad(ur),grad(z_err_real[i])) * dx) /  assemble(inner(ur,z_err_real[i]) * dx) - lam_p)
    #print(eta_h_predict)

    z = z_err_real[i]  # freeze the reference for this iteration
    A = assemble(inner(grad(ur), grad(z)) * dx)
    B = assemble(inner(ur, z) * dx)

    # Optional: promote to plain Python floats/complex to avoid array wrappers
    A = complex(A)  # or float(A) if you’re sure it's real
    B = complex(B)
    eta1 = abs((A - lam_p*B) / (B))
    eta2 = abs(A / B - lam_p)

    #print(f"A={A}, B={B}")
    #print(f"eta1={eta1}, eta2={eta2}, diff={eta1-eta2}")
    
    with eta_abs.dat.vec_ro as v:         # read-only PETSc Vec
        eta_array = v.getArray().copy()   # NumPy array copy
    total = eta_array.sum() 
    eta = abs(lam[i] - exact_lambdas[i])
    #print("Exact lam = ", exact_lambdas[i], " Dual lam = ", lam_d[i], "Primal_lam = ", lam[i])
    #print("Actual eta_h = ", abs(lam_d[i] - lam[i]))
    print("Sum(eta_T) =", total, "eta_h = ", eta1, " eta = ", eta) 
