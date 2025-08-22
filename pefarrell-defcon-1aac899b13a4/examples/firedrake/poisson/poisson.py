# -*- coding: utf-8 -*-
from firedrake import *
from defcon import *

# Solve the Poisson equation.
# This is to test the DWR error estimators.

class PoissonProblem(BifurcationProblem):
    def mesh(self, comm):
        return UnitSquareMesh(8, 8, quadrilateral=True, comm=comm)

    def function_space(self, mesh):
        return FunctionSpace(mesh, "CG", 1)

    def parameters(self):
        delta = Constant(0)
        return [(delta, "delta", r"$\delta$")]

    def residual(self, u, params, v):
        (x, y) = SpatialCoordinate(u.function_space().mesh())
        u_exact = 256*(1-x)*x*(1-y)*y*exp(-((x-0.5)**2+(y-0.5)**2)/10)
        f = -div(grad(u_exact))

        F = (
              inner(grad(v), grad(u))*dx
            - inner(v, f)*dx
            )

        return F

    def boundary_conditions(self, V, params):
        bcs = [DirichletBC(V, 0, "on_boundary")]
        return bcs

    def functionals(self):
        def normal_gradient(u, params):
            n = FacetNormal(u.function_space().mesh())
            return assemble(dot(grad(u), n)*ds)

        return [(normal_gradient, "normal_gradient", r"$\int_{\partial \Omega} \nabla u \cdot n \ \mathrm{d}s$", lambda u, params: dot(grad(u), FacetNormal(u.function_space().mesh()))*ds)]

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, V, params, n):
        return Function(V)

    def number_solutions(self, params):
        return 1

    def solver_parameters(self, params, task, **kwargs):
        params = {
            "snes_max_it": 100,
            "snes_atol": 1.0e-9,
            "snes_rtol": 0.0,
            "snes_monitor": None,
            "snes_linesearch_type": "basic",
            "snes_linesearch_maxstep": 1.0,
            "snes_linesearch_damping": 1.0,
            "mat_type": "aij",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps"
        }
        return params

    def estimate_error(self, *args, **kwargs):
        return estimate_error_dwr(self, *args, **kwargs)

if __name__ == "__main__":
    dc = DeflatedContinuation(problem=PoissonProblem(), teamsize=1, verbose=True, profile=False, clear_output=True)
    dc.run(values={"delta": 0})
