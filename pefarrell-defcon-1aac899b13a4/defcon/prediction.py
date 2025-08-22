# Implement tangent predictor for continuation steps. These
# are intended for use in the BifurcationProblem.predict method.

import defcon.backend as backend
from defcon.newton import newton
from defcon.tasks import TangentPredictionTask

def tangent(problem, solution, oldparams, newparams, hint=None):
    coldparams = [backend.Constant(x) for x in oldparams]
    chgparams = [backend.Constant(new-old) for (new, old) in zip(newparams, oldparams)]

    Z = solution.function_space()
    v = backend.TestFunction(Z)
    w = backend.TrialFunction(Z)

    # FIXME: cache the symbolic calculation once, it can be expensive sometimes
    du = backend.Function(Z)

    F = problem.residual(solution, coldparams, v)
    G = backend.derivative(F, solution, du) + sum(
        backend.derivative(F, oldparam, chgparam)
        for (oldparam, chgparam) in zip(coldparams, chgparams)
    )

    J = problem.jacobian(G, du, chgparams, v, w)

    # FIXME: figure out if the boundary conditions depend on
    # the parameters, and set the boundary conditions on the update
    dubcs = problem.boundary_conditions(Z, newparams)
    if dubcs:
        [dubc.homogenize() for dubc in dubcs]

    dm = problem._dm

    # FIXME: make this optional
    teamno = 0

    task = TangentPredictionTask(oldparams, newparams)

    # FIXME: there's probably a more elegant way to do this.
    # Or should we use one semismooth Newton step? After all
    # we already have a Newton linearisation.
    vi = "bounds" in problem.__class__.__dict__
    if vi:
        # If we're dealing with a VI, we need to enforce the appropriate
        # bound constraints on the update problem. Essentially, we need
        # that
        # lb <= u + du <= ub
        # so
        # lb - u <= du <= ub - u

        class FixTheBounds(object):
            def bounds(self, Z, params, initial_guess):
                (lb, ub) = problem.bounds(Z, newparams, initial_guess)
                lb.vector().axpy(-1.0, solution.vector())
                ub.vector().axpy(-1.0, solution.vector())

                return (lb, ub)

            def __getattr__(self, attr):
                return getattr(problem, attr)

        newproblem = FixTheBounds()
    else:
        newproblem = problem

    (success, iters) = newton(G, J, du, dubcs,
                              chgparams,
                              newproblem,
                              problem.solver_parameters(oldparams, task),
                              teamno, deflation=None, dm=dm)


    if not success:
        # Should we raise an Exception here? After all, this is only an auxiliary
        # problem.
        backend.warning("Tangent linearisation failed")

    solution.assign(solution + du)

    return None


def secant(problem, solution, oldparams, newparams, hint=None):
    if hint is None:
        # First time we hit the secant continuation, nothing to do
        problem.__secant_solution = backend.Function(solution)
        return tuple(oldparams)
    else:
        prevparams = hint
        prevsolution = problem.__secant_solution
        oldsolution = backend.Function(solution)  # for optimisation: don't malloc each time

        # Find out which parameter has varied, and make sure they're all distinct
        for (i, prevparam, oldparam) in zip(range(len(oldparams)), prevparams, oldparams):
            if prevparam != oldparam:
                newparam = newparams[i]
                break

        assert oldparam != prevparam
        assert newparam != oldparam
        assert prevparam != newparam

        # Calculate the line joining (prevparam, prevsolution) and (oldparam, oldsolution)
        m = (oldsolution - prevsolution) / (oldparam - prevparam)  # slope
        solution.assign(m * (newparam - oldparam) + oldsolution)   # extrapolate line to newparam

        problem.__secant_solution.assign(oldsolution)  # update prevsolution with oldsolution
        return tuple(oldparams)  # hint for next time
