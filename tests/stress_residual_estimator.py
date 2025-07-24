from firedrake import *
import numpy

nx = 40
mesh = UnitSquareMesh(nx, nx)
degree = 1
S = VectorFunctionSpace(mesh, "BDM", degree)
V = VectorFunctionSpace(mesh, "DG", degree-1)
Q = FunctionSpace(mesh, "CG", degree)

# Mixed Function Space
Z = S * V * Q

print(S.value_shape, S.value_size)
print(V.value_shape, V.value_size)
print(Q.value_shape, Q.value_size)
print(Z.value_shape, Z.value_size)

test = TestFunction(Z)
z = Function(Z)

# symbolic split
sigma, u, gamma = split(z)
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

# LHS〈Aσ, τ 〉 + 〈div σ, v〉 + 〈u, div τ 〉 + 〈σ, η〉 + 〈γ, τ〉
# RHS〈g, v〉 + 〈u0, τ · n〉∂Ω



x, y = SpatialCoordinate(mesh)

uexact = as_vector([x*y*sin(pi*y), 0])
sigma_exact = Ainv(sym(grad(uexact)))
g = div(sigma_exact)

u0 = uexact
n = FacetNormal(mesh)

F = (inner(A(sigma), tau)*dx
     + inner(u, div(tau))*dx
     + inner(gamma, skw(tau))*dx
     + inner(div(sigma), v)*dx
     + inner(skw(sigma), eta)*dx
     - inner(g, v)*dx
     - inner(u0, dot(tau, n))*ds
     )



#bcs = [DirichletBC(Z.sub(0), 0, (3,4))]
bcs = []

solve(F == 0, z, bcs=bcs)

# Goal Functional
psi = y * (y-1)
M = inner(dot(sigma, n), as_vector([psi, 0]))*ds(2)
print("goal functional", assemble(M))


# numerical split
sigma, u, gamma = z.subfunctions
sigma.rename("sigma")
u.rename("u")
gamma.rename("gamma")


print("error u", errornorm(uexact, u) / norm(uexact))
print("error sigma", errornorm(sigma_exact, sigma)/ norm(sigma_exact))
print("error skw(sigma)", norm(skw(sigma)))


file = VTKFile("output/stress.pvd")
file.write(sigma, u, gamma)

exit()

dim = 2

variant = "integral"
B = FunctionSpace(mesh, "B", dim+1, variant=variant)

R = VectorFunctionSpace(mesh, "DG", degree, dim=Z.value_size)

element = BrokenElement(FiniteElement("FB", degree+dim))