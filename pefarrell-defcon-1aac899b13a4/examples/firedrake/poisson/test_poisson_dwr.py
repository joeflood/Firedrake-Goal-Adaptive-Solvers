from defcon import *
from defcon.newton import newton
from firedrake import *
from mpi4py import MPI

from firedrake.petsc import PETSc
PETSc.Sys.popErrorHandler()

import poisson


def test_poisson_dwr():
    problem = poisson.PoissonProblem()
    params = Constant((0,))

    mesh = problem.mesh(MPI.COMM_WORLD)
    Z = problem.function_space(mesh)

    z = problem.initial_guess(Z, params, 0)

    v = TestFunction(Z)
    w = TrialFunction(Z)

    F = problem.residual(z, params, v)
    J = problem.jacobian(F, z, params, v, w)
    bcs = problem.boundary_conditions(Z, params)

    task = DeflationTask(0, None, None, 0, params)
    teamno = 0

    sp = problem.solver_parameters(params, task)

    (success, iters) = newton(F, J, z, bcs,
                              params,
                              problem,
                              sp,
                              teamno)

    assert success

    functionals = problem.functionals()[0]
    Jcoarse = functionals[0](z, params)

    Jest = problem.estimate_error(F, functionals[3](z, params), z, bcs, params)

    # Now we hardcode knowledge about what the true functional is.
    (x, y) = SpatialCoordinate(mesh)
    z_exact = 256*(1-x)*x*(1-y)*y*exp(-((x-0.5)**2+(y-0.5)**2)/10)
    n = FacetNormal(mesh)
    Jtrue = assemble(dot(grad(z_exact), n)*ds)

    effectivity = Jest / (Jcoarse - Jtrue)

    print(f"Functional approximation: {Jcoarse}")
    print(f"True functional: {Jtrue}")
    print(f"Error estimator: {Jest}")
    print(f"Effectivity: {effectivity}")
    print(f"Corrected estimate: {Jcoarse - Jest}")
    print(f"Error in corrected estimate: {Jcoarse - Jest - Jtrue}")

    assert 0.98 <= effectivity <= 1.02


if __name__ == "__main__":
    test_poisson_dwr()
