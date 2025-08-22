from defcon import *
from dolfin import *

class BratuProblem(BifurcationProblem):
    def mesh(self, comm):
        return IntervalMesh(comm, 400, 0, 1)

    def function_space(self, mesh):
        V = FunctionSpace(mesh, "CG", 2)
        return V

    def parameters(self):
        lmbda = Constant(0)

        return [(lmbda, "lambda", r"$\lambda$")]

    def residual(self, u, params, v):
        lmbda = params[0]

        F = (
              - inner(grad(u), grad(v))*dx
              + lmbda*exp(u)*v*dx
            )

        return F

    def boundary_conditions(self, V, params):
        return [DirichletBC(V, 0.0, "on_boundary")]

    def functionals(self):
        def eval(u, params):
            j = u((0.5,))
            return j

        return [(eval, "eval", r"$u(0.5)$")]

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, V, params, n):
        return interpolate(Constant(0), V)

    def number_solutions(self, params):
        lmbda = params[0]
        if lmbda > 3.513: return 0
        else: return 2

    def solver_parameters(self, params, task, **kwargs):
        return {
               "snes_max_it": 100,
               "snes_atol": 1.0e-9,
               "snes_rtol": 0.0,
               "snes_monitor": None,
               "ksp_type": "preonly",
               "pc_type": "lu"
               }

    def predict(self, problem, solution, oldparams, newparams, hint):
        # Use tangent continuation
        return tangent(problem, solution, oldparams, newparams, hint)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dc = DeflatedContinuation(problem=BratuProblem(), teamsize=1, verbose=True, clear_output=True)
    dc.run(values={"lambda": list(arange(0.0, 3.6, 0.01)) + [3.6]})

    dc.bifurcation_diagram("eval")
    plt.title(r"The Bratu problem")
    plt.savefig("bifurcation.pdf")
