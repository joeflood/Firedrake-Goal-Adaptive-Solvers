from defcon import *
from firedrake import *

L = 10 # length of domain

class PainleveProblem(BifurcationProblem):
    def mesh(self, comm):
        return IntervalMesh(400, -L, 0, comm=comm)

    def function_space(self, mesh):
        return FunctionSpace(mesh, "CG", 2)

    def parameters(self):
        a = Constant(0)
        return [(a, "a", r"$a$")]

    def residual(self, y, params, v):
        a = params[0]

        x = SpatialCoordinate(y.function_space().mesh())[0]

        F = (
             - inner(grad(y), grad(v))*dx
             - inner(a*y*y, v)*dx 
             - inner(x, v)*dx
            )

        return F

    def boundary_conditions(self, V, params):
        bcs = [DirichletBC(V, Constant(sqrt(L/6.0)), 1),
               DirichletBC(V, Constant(0),           2)]
        return bcs

    def functionals(self):
        def sqL2(y, params):
            j = assemble(y*y*dx)
            return j

        return [(sqL2, "sqL2", r"$\|y\|^2$")]

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, V, params, n):
        m = -sqrt(1.0/(6*L))
        x = SpatialCoordinate(V.mesh())[0]
        return Function(V).interpolate(m*x)

    def number_solutions(self, params):
        return 2

    def solver_parameters(self, params, task, **kwargs):
        return {
               "snes_type": "newtonls",
               "snes_linesearch_type": "basic",
               "snes_linesearch_damping": 0.1,
               "snes_max_it": 10000,
               "snes_max_funcs": 10000,
               "snes_monitor": None,
               "snes_linesearch_monitor": None,
               "snes_converged_reason": None,
               "snes_stol": 0.0,
               "ksp_type": "preonly",
               "pc_type": "lu",
               }

if __name__ == "__main__":
    problem = PainleveProblem()
    deflation = ShiftedDeflation(problem, power=2, shift=1)
    dc = DeflatedContinuation(problem, deflation=deflation, teamsize=1, verbose=True, clear_output=True)
    dc.run(values={"a": [6.0]})
