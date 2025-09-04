from firedrake import *
from netgen.occ import *
import sys
from goal_adaptivity import GoalAdaptiveNonlinearVariationalSolver

sys.path.insert(0, "./algorithm")
nx = 10
mesh = Mesh(unit_square.GenerateMesh(maxh=1/nx))
degree = 1

# Define solver parameters ---------------------
solver_parameters = {
    "max_iterations": 20,
    "output_dir": "output_final",
    "run_name": "elasticity_final",
    "manual_indicators": False,
    "dual_extra_degree": 1,
    "use_adjoint_residual": False,
    "primal_low_method": "solve",
    "dual_low_method": "interpolate"
    #"uniform_refinement": True
    #"use_adjoint_residual": True
}

# Define actual problem -----------------------
# Define function spaces
S = VectorFunctionSpace(mesh, "BDM", degree)
V = VectorFunctionSpace(mesh, "DG", degree-1)
Q = FunctionSpace(mesh, "CG", degree)

# Mixed Function Space
T = S * V * Q

# Mixed test function
test = TestFunction(T)
t = Function(T)

# symbolic split
sigma, u, gamma = split(t)
tau, v, eta = split(test)

dim = mesh.geometric_dimension()

mu = Constant(1)
lam = Constant(100)
I = Identity(dim)

def A(sigma):
    return (1/(2*mu)) * (sigma - (lam/(2*mu+dim*lam)) * tr(sigma)*I)
def Ainv(sigma):
    return (2*mu*sigma + lam * tr(sigma)*I)
def skw(sigma):
    return sigma[0, 1] - sigma[1, 0]

x, y = SpatialCoordinate(mesh)

u_exact = as_vector([x*y*sin(pi*y), 0])
sigma_exact = Ainv(sym(grad(u_exact)))
gamma_exact = 0.5*(grad(u_exact)[0,1] - grad(u_exact)[1,0])
exact_sol = [sigma_exact, u_exact, gamma_exact]

g = div(sigma_exact)

u0 = u_exact
n = FacetNormal(mesh)

# F =〈Aσ, τ 〉 + 〈div σ, v〉 + 〈u, div τ 〉 + 〈σ, η〉 + 〈γ, τ〉 -〈g, v〉 - 〈u0, τ · n〉∂Ω
F = (inner(A(sigma), tau)*dx
    + inner(u, div(tau))*dx
    + inner(gamma, skw(tau))*dx
    + inner(div(sigma), v)*dx
    + inner(skw(sigma), eta)*dx
    - inner(g, v)*dx
    - inner(u0, dot(tau, n))*ds
    )

bcs = []

# Goal Functional
psi = y * (y-1)
M = inner(dot(sigma, n), as_vector([0, psi]))*ds(2)
tolerance = 0.00001


problem = NonlinearVariationalProblem(F, t, bcs)

adaptive_problem = GoalAdaptiveNonlinearVariationalSolver(problem, M, tolerance, solver_parameters, exact_solution=exact_sol)

adaptive_problem.solve()


