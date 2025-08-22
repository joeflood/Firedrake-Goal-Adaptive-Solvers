from defcon import *
from firedrake import *

class YamabeProblem(BifurcationProblem):
    def mesh(self, comm):
        # With order=2, we represent the geometry exactly
        return OpenCascadeMeshHierarchy("mesh/disk.step", element_size=10, levels=1, order=2, comm=comm)[-1]

    def function_space(self, mesh):
        return FunctionSpace(mesh, "CG", 2)

    def parameters(self):
        a = Constant(0)
        return [(a, "a", r"$a$")]

    def residual(self, y, params, v):
        a = params[0]

        x = SpatialCoordinate(y.function_space().mesh())
        r = sqrt(x[0]**2 + x[1]**2)
        rho = 1.0/r**3

        F = (
             a*inner(grad(y), grad(v))*dx +
             rho * inner(y**5, v)*dx +
             Constant(-1.0/10.0)*inner(y, v)*dx
            )

        return F

    def boundary_conditions(self, V, params):
        bcs = [DirichletBC(V, Constant(1.0), "on_boundary")]
        return bcs

    def functionals(self):
        def sqL2(y, params):
            j = assemble(y*y*dx)
            return j

        return [(sqL2, "sqL2", r"$\|y\|^2$")]

    def number_initial_guesses(self, params):
        return 2

    def initial_guess(self, V, params, n):
        return Function(V).interpolate(Constant((-1)**n))

    def solver_parameters(self, params, task, **kwargs):
        return {
               "snes_max_it": 100,
               "snes_stol": 0.0,
               "snes_rtol": 1.0e-10,
               "snes_monitor": None,
               "snes_divergence_tolerance": -1,
               "snes_linesearch_type": "basic",
               "snes_linesearch_maxstep": 1,
               "snes_linesearch_monitor": None,
               "ksp_monitor": None,
               "ksp_type": "gcr",
               "ksp_gmres_restart": 100,
               "ksp_max_it": 1000,
               "ksp_atol": 1.0e-10,
               "ksp_rtol": 1.0e-10,
               "pc_type": "mg",
               }

if __name__ == "__main__":
    problem = YamabeProblem()
    deflation = ShiftedDeflation(problem, power=1, shift=1.0e-2)
    dc = DeflatedContinuation(problem, deflation=deflation, teamsize=1, verbose=True, clear_output=True)
    dc.run(values={"a": [8.0]})
