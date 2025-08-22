from defcon import *
from firedrake import *

class BratuProblem(BifurcationProblem):
    def mesh(self, comm):
        return UnitIntervalMesh(400, comm=comm)

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
        def sqL2(y, params):
            j = assemble(y*y*dx)
            return j

        def eval(u, params):
            j = u((0.5,))
            return float(j)

        return [(sqL2, "sqL2", r"$\|y\|^2$", lambda y, params: y*y*dx),
                (eval, "eval", r"$u(0.5)$")]

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, V, params, n):
        return Function(V)

    def number_solutions(self, params):
        lmbda = params[0]
        if lmbda > 3.513: return 0
        else: return 2

    def solver_parameters(self, params, task, **kwargs):
        if isinstance(task, ArclengthTask):
            return {
                "mat_type": "matfree",
                "snes_type": "newtonls",
                "snes_monitor": None,
                "snes_converged_reason": None,
                #"snes_atol": 1.0e-6,
                "snes_linesearch_type": "basic",
                "ksp_type": "fgmres",
                "ksp_monitor_true_residual": None,
                "ksp_max_it": 10,
                "pc_type": "fieldsplit",
                "pc_fieldsplit_type": "schur",
                "pc_fieldsplit_schur_fact_type": "full",
                "pc_fieldsplit_0_fields": "0",
                "pc_fieldsplit_1_fields": "1",
                "fieldsplit_0_ksp_type": "preonly",
                "fieldsplit_0_pc_type": "python",
                "fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
                "fieldsplit_0_assembled_pc_type": "lu",
                "fieldsplit_0_assembled_pc_factor_mat_solver_type": "mumps",
                "fieldsplit_1_ksp_type": "gmres",
                "fieldsplit_1_ksp_monitor_true_residual": None,
                "fieldsplit_1_ksp_max_it": 1,
                "fieldsplit_1_ksp_convergence_test": "skip",
                "fieldsplit_1_pc_type": "none",
                }
        else:
            return {
                "snes_max_it": 100,
                "snes_atol": 1.0e-9,
                "snes_rtol": 0.0,
                "snes_monitor": None,
                "snes_linesearch_type": "basic",
                "ksp_type": "preonly",
                "pc_type": "lu"
                }

    def predict(self, problem, solution, oldparams, newparams, hint):
        # Use tangent continuation
        return tangent(problem, solution, oldparams, newparams, hint)

    def estimate_error(self, *args, **kwargs):
        return estimate_error_dwr(self, *args, **kwargs)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dc = DeflatedContinuation(problem=BratuProblem(), teamsize=1, verbose=True, clear_output=True)
    dc.run(values={"lambda": list(arange(0.0, 3.6, 0.01)) + [3.6]})

    dc.bifurcation_diagram("eval")
    plt.title(r"The Bratu problem")
    plt.savefig("bifurcation.pdf")
