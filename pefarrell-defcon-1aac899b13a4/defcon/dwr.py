# A simple example of a dual-weighted residual error estimator.

from defcon.backend import Function, TestFunction, TrialFunction, action, adjoint, derivative, solve, assemble, Constant
from defcon.tasks import AdjointTask
import ufl.algorithms
import ufl
from ufl import replace

def estimate_error_dwr(bifurcationproblem, F, J, state, bcs, params):
    V = state.function_space()
    v = ufl.algorithms.extract_arguments(F)[0]

    # Get an enriched function space for solving the dual problem.
    # Defaults to increasing the polynomial degree.
    Vf = bifurcationproblem.enrich_function_space(V)

    z = Function(Vf)  # dual solution
    vz = TestFunction(Vf)

    # Set up adjoint residual
    G = action(adjoint(derivative(F, state, TrialFunction(Vf))), z) - derivative(J, state, v)
    G = replace(G, {v: vz})

    # Homogenise and promote the boundary conditions
    hbcs = [bc.reconstruct(V=Vf, g=ufl.zero(Vf.value_shape)) for bc in bcs]

    # Get solver parameters for adjoint problem
    sp = bifurcationproblem.solver_parameters(params, AdjointTask(params))

    # Solve the adjoint equation
    solve(G == 0, z, hbcs, solver_parameters=sp)

    # Evaluate the error estimate: assemble
    # the primal residual at the dual solution
    Jest = assemble(replace(F, {v: z}))

    return Jest
