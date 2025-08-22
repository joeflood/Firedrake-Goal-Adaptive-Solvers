# -*- coding: utf-8 -*-
from firedrake import *
from defcon import *

class ObstacleProblem(BifurcationProblem):
    def mesh(self, comm):
        return UnitSquareMesh(64, 64, comm=comm)

    def function_space(self, mesh):
        V = FunctionSpace(mesh, "CG", 1)
        return V

    def parameters(self):
        f = Constant(0)
        scale = Constant(0)
        return [(f, "f", r"$f$"),
                (scale, "scale", "scale")]

    def residual(self, u, params, v):
        f = params[0]

        F = (
              inner(grad(u), grad(v))*dx
            - inner(f, v)*dx
            )

        return F

    def boundary_conditions(self, V, params):
        return [DirichletBC(V, 0, "on_boundary")]

    def functionals(self):
        def uL2(u, params):
            j = assemble(u*u*dx)
            return j

        return [(uL2, "uL2", r"$\|u\|^2$")]

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, V, params, n):
        return Function(V)

    def number_solutions(self, params):
        return 1

    def solver_parameters(self, params, task, **kwargs):
        return {
               "snes_max_it": 50,
               "snes_atol": 1.0e-9,
               "snes_rtol": 1.0e-9,
               "snes_monitor": None,
               "ksp_type": "preonly",
               "ksp_monitor": None,
               "ksp_rtol": 1.0e-10,
               "ksp_atol": 1.0e-10,
               "pc_type": "lu",
               "pc_factor_mat_solver_type": "mumps",
               }

    def bounds(self, V, params, initial_guess):
        scale = params[1]
        x = SpatialCoordinate(V.mesh())[0]
        obstacle = conditional(lt(x, +0.25), scale*-0.2,
                   conditional(lt(x, +0.50), scale*-0.4,
                   conditional(lt(x, +0.75), scale*-0.6,
                                             scale*-0.8)))

        l = Function(V).interpolate(obstacle)
        u = Function(V).interpolate(Constant(1e20))
        return (l, u)

    def save_pvd(self, u, pvd, params):
        (l, _) = self.bounds(u.function_space(), params, None)
        l.rename("LowerBound")
        u.rename("Solution")
        pvd.write(u, l)

    def predict(self, problem, solution, oldparams, newparams, hint):
        # This actually does all the work of the solver.
        # The linearised prediction is essentially perfect because the
        # energy to be minimised is quadratic.
        return tangent(problem, solution, oldparams, newparams, hint)

if __name__ == "__main__":
    problem = ObstacleProblem()
    dc = DeflatedContinuation(problem=problem, teamsize=1, verbose=True, clear_output=True)
    dc.run(values={"f": linspace(0, -20, 41), "scale": 1}, freeparam="f")
