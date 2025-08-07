from firedrake import *
from netgen.occ import *
import sys
from algorithm import *
from functools import singledispatch
from firedrake.mg.ufl_utils import coarsen
from adaptive_mg.adaptive import AdaptiveMeshHierarchy
from adaptive_mg.adaptive_transfer_manager import AdaptiveTransferManager

nx = 3
mesh = Mesh(unit_square.GenerateMesh(maxh=1/nx))
degree = 1

meshctx = MeshCtx(mesh)

# Define solver parameters ---------------------
solver_parameters = {
    "degree": 1,
    "dual_solve_method": "high_order",
    "dual_solve_degree": "degree + 1",
    "residual_solve_method": "automatic",
    "residual_degree": "degree",
    "dorfler_alpha": 0.5,
    "goal_tolerance": 0.00001,
    "max_iterations": 10,
    "output_dir": "output/elasticity",
    "write_at_iteration": True
}

solverctx = SolverCtx(solver_parameters)

mesh = meshctx.mesh

# Define function spaces
S = VectorFunctionSpace(mesh, "BDM", degree)
V = VectorFunctionSpace(mesh, "DG", degree-1)
Q = FunctionSpace(mesh, "CG", degree)

# Mixed Function Space
T = S * V * Q

#for i, sub in enumerate(T):
#   elem = sub.ufl_element()          # UFL element object
#    print(f"Subspace {i}:")
#    print(f"  Shape       : {sub.value_shape}")             # value shape (e.g. (), (2,), (3,))
#    print(f"  Element type: {elem.family()}")         # e.g. "BDM", "DG", "CG"
#    print(f"  Degree      : {elem.degree()}")         # polynomial degree

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

u_f = t.sub(2)
u_f.interpolate((sin(pi*x)*cos(pi*y)))

markers_space = FunctionSpace(mesh, "DG", 0)
markers = Function(markers_space)
markers.assign(1.0)

new_mesh = mesh.refine_marked_elements(markers)

amh = AdaptiveMeshHierarchy([mesh])
atm = AdaptiveTransferManager()
amh.add_mesh(new_mesh)

coef_map = {}
F_new = coarsen(F, coarsen, coefficient_mapping=coef_map)

t_new = coarsen(t, coarsen, coefficient_mapping=coef_map)
t_new = ceof_map[t]
sigma_new, u_new, gamma_new = split(t_new)

print(sigma)
print(sigma_new)

u_f = t.sub(2)
u_new_f = t_new.sub(2)

print(u_f.dat.data_ro)
print(u_new_f.dat.data_ro)

print("Writing... ")
VTKFile("coarse.pvd").write(u_f)
VTKFile("fine.pvd").write(u_new_f)


'''
@singledispatch
def coarsen_override(expr, self, coefficient_mapping=None):
    return coarsen(expr, self, coefficient_mapping=None)

@coarsen_override.register(ufl.Mesh)
def coarsen_mesh(mesh, target_mesh, coefficient_mapping=None):
    return target_mesh



def make_coarsener(target_mesh):
    """Return a `coarsen` variant that always maps every mesh to *target_mesh*."""

    @singledispatch
    def c(expr, self, coefficient_mapping=None):
        # delegate all un-overridden cases to the original implementation,
        # but recurse with *this* dispatcher (`c`) so the override is used
        return coarsen(expr, c, coefficient_mapping=coefficient_mapping)

    # copy every existing registration except the one for `ufl.Mesh`
    for typ, fn in coarsen.registry.items():
        if typ is not ufl.Mesh and typ is not object:
            c.register(typ)(fn)

    # override only the mesh rule
    @c.register(ufl.Mesh)
    def _(mesh, self, coefficient_mapping=None):     # noqa: F811
        return target_mesh

    return c

my_coarsen    = make_coarsener(new_mesh)
'''
#F_new   = my_coarsen(F, my_coarsen) 

def form_on_new_mesh(F_old, mesh_new):
    # rebuild the function space on new mesh
    element = T.ufl_element()
    V_new = FunctionSpace(mesh_new, element)

    # remap every Argument in the form
    arg_map = {a: a.reconstruct(V_new) for a in F_old.arguments()}

    # remap Functions that appear in the form
    func_map = {}
    for C in F_old.coefficients():
        if isinstance(C, Function):
            V_new = C.function_space().reconstruct(mesh_new)
            C_new = Function(V_new, name=C.name())
            func_map[C] = C_new

    F_mid = replace(F_old, {**arg_map, **func_map})
    
    # remap integrals
    new_integrals = [I.reconstruct(domain=mesh_new) for I in F_mid.integrals()]
    F_new = ufl.Form(new_integrals)
    
    return F_new
