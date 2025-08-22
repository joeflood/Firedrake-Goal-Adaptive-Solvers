# Just solve a PDE specified in defcon format
# for a given set of parameters.

import defcon.backend as backend
from defcon.newton import newton
from defcon.tasks import DeflationTask
from defcon.mg import create_dm


def dcsolve(problem, params, comm, guess=None, deflation=None, sp=None):

    if isinstance(guess, backend.Function):
        mesh = guess.function_space().mesh()
        Z = guess.function_space()
        z = guess.copy(deepcopy=True)
    else:
        mesh = problem.mesh(comm)
        Z = problem.function_space(mesh)
        z = problem.initial_guess(Z, params, guess or 0)

    v = backend.TestFunction(Z)
    w = backend.TrialFunction(Z)

    F = problem.residual(z, params, v)
    J = problem.jacobian(F, z, params, v, w)
    bcs = problem.boundary_conditions(Z, params)

    task = DeflationTask(0, None, None, 0, params)
    dm = create_dm(Z, problem)
    teamno = 0  # FIXME: make this optional

    if sp is None:
        sp = problem.solver_parameters(params, task)

    (success, iters) = newton(F, J, z, bcs,
                              params,
                              problem,
                              sp,
                              teamno, deflation=deflation, dm=dm)

    if success:
        return (success, iters, z)
    else:
        return (success, iters, None)
