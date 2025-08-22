from __future__ import absolute_import

from defcon.bifurcationproblem import BifurcationProblem
from defcon.newton import compute_tau
from defcon.backendimporter import get_deep_submat
from defcon.backend import *
from petsc4py import PETSc
from numpy import where, array, int32
from ufl import diag

def fb(a, b):
    """
    Fischer--Burmeister merit function.
    """
    return sqrt(a**2 + b**2) - a - b

class ComplementarityProblem(BifurcationProblem):
    r"""
    A class for finite-dimensional nonlinear complementarity problems, i.e. find x st

    0 <= x \perp F(x) >= 0

    Possible generalisation: to MCPs instead of NCPs.
    """

    def __init__(self, F, N):
        self.F = F
        self.N = N

    def mesh(self, comm):
        mesh = UnitIntervalMesh(comm, 1)
        return mesh

    def function_space(self, mesh):
        Re = FiniteElement("R", interval, 0)
        Ve = MixedElement([Re]*self.N)
        V = FunctionSpace(mesh, Ve)

        return V

    def residual(self, z, params, v):

        f = self.F(z, params)
        Psi = sum(inner(v[i], fb(z[i], f[i]))*dx for i in range(self.N))

        return Psi

    def jacobian(self, Psi, z, params, v, dz):

        f = as_vector(self.F(z, params))
        df = derivative(f, z, dz)
        N = self.N

        sqrts = [sqrt(z[i]**2 + f[i]**2) for i in range(N)]
        safe  = Constant(1.0/sqrt(2)) - Constant(1)

        I_coeffs = [conditional(gt(sqrts[i], 0), z[i]/sqrts[i] - 1, safe) for i in range(N)]
        D_I = diag(as_vector(I_coeffs))

        J_coeffs = [conditional(gt(sqrts[i], 0), f[i]/sqrts[i] - 1, safe) for i in range(N)]
        D_J = diag(as_vector(J_coeffs))

        J = (
            + inner(v, dot(D_I, dz))*dx    # diagonal matrix times identity
            + inner(v, dot(D_J, df))*dx    # diagonal matrix times problem Jacobian
            )

        return J

    def boundary_conditions(self, V, params):
        return []

    def functionals(self):
        def fetch_component(i):
            def func(z, params):
                return z.vector().get_local()[i]
            return (func, "z[%d]" % i, r"z_{%d}" % i)

        def l2norm(z, params):
            return z.vector().norm("l2")

        return [fetch_component(i) for i in range(self.N)] + [(l2norm, "l2norm", r"$\|z\|$")]

    def squared_norm(self, a, b, params):
        return inner(a - b, a - b)*dx

