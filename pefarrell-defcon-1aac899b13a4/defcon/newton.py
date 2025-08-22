from __future__ import absolute_import

from petsc4py import PETSc
from numpy import isnan

import sys
import weakref

import defcon.backend as backend


# Unfortunately DOLFIN and firedrake are completely different in how they
# do the solve. So we have to branch based on backend here.

if backend.__name__ == "dolfin":
    from defcon.backend import PETScOptions, PETScVector, as_backend_type

    def getEdy(deflation, y, dy, vi_inact):
        deriv = as_backend_type(deflation.derivative(y)).vec()
        if vi_inact is not None:
            deriv_ = deriv.getSubVector(vi_inact)
        else:
            deriv_ = deriv

        out = -deriv_.dot(dy)

        if vi_inact is not None:
            deriv.restoreSubVector(vi_inact, deriv_)

        return out

    def setSnesMonitor(prefix):
        PETScOptions.set(prefix + "snes_monitor_cancel")
        PETScOptions.set(prefix + "snes_monitor")

    def setSnesBounds(snes, bounds):
        (lb, ub) = bounds
        snes.setVariableBounds(backend.as_backend_type(lb.vector()).vec(), backend.as_backend_type(ub.vector()).vec())

elif backend.__name__ == "firedrake":
    from defcon.backend import NonlinearVariationalSolver
    from defcon.backend.petsc import PETSc

    def getEdy(deflation, y, dy, vi_inact):

        with deflation.derivative(y).dat.vec as deriv:
            if vi_inact is not None:
                deriv_ = deriv.getSubVector(vi_inact)
            else:
                deriv_ = deriv

            out = -deriv_.dot(dy)

            if vi_inact is not None:
                deriv.restoreSubVector(vi_inact, deriv_)

        return out

    def setSnesMonitor(prefix):
        opts = PETSc.Options()
        opts.setValue(prefix + "snes_monitor_cancel", "")
        opts.setValue(prefix + "snes_monitor", "")

    def setSnesBounds(snes, bounds):
        (lb, ub) = bounds
        with lb.dat.vec_ro as lb_, ub.dat.vec_ro as ub_:
            snes.setVariableBounds(lb_, ub_)

else:
    raise ImportError("Unknown backend")

if hasattr(backend, 'ConvergenceError'):
    from defcon.backend import ConvergenceError
else:
    class ConvergenceError(Exception):
        pass

def compute_tau(deflation, state, update_p, vi_inact):
    if deflation is not None:
        Edy = getEdy(deflation, state, update_p, vi_inact)

        minv = 1.0 / deflation.evaluate(state)
        tau = 1/(1 - minv*Edy)
        return tau
    else:
        return 1

class DeflatedKSP(object):
    def __init__(self, deflation, y, ksp, snes):
        self.deflation = deflation
        self.y = y
        self.ksp = ksp
        self.snes = weakref.proxy(snes)

    def solve(self, ksp, b, dy_pet):
        # Use the inner ksp to solve the original problem
        self.ksp.setOperators(*ksp.getOperators())
        self.ksp.solve(b, dy_pet)
        deflation = self.deflation

        if self.snes.getType().startswith("vi"):
            vi_inact = self.snes.getVIInactiveSet()
        else:
            vi_inact = None

        tau = compute_tau(deflation, self.y, dy_pet, vi_inact)
        dy_pet.scale(tau)

        ksp.setConvergedReason(self.ksp.getConvergedReason())

    def reset(self, ksp):
        self.ksp.reset()

    def view(self, ksp, viewer):
        self.ksp.view(viewer)

def newton(F, J, y, bcs, params, problem, solver_params,
           teamno, deflation=None, dm=None, prefix=""):
    comm = y.function_space().mesh().mpi_comm()
    npproblem = problem.nonlinear_problem(F, J, y, bcs)
    npproblem.deflation = deflation

    solver = problem.solver(npproblem, params, solver_params, prefix=prefix, dm=dm, deflation=deflation)
    snes = solver.snes

    snes.incrementTabLevel(teamno*2)
    setSnesMonitor(prefix)

    vi = "bounds" in problem.__class__.__dict__
    if vi:
        snes.setType("vinewtonrsls")
        bounds = problem.bounds(y.function_space(), params, y)
        setSnesBounds(snes, bounds)

    fiddle_ksp = True
    if snes.ksp.type == "python":
        if isinstance(snes.ksp.getPythonContext(), DeflatedKSP):
            fiddle_ksp = False

    if fiddle_ksp:
        oldksp = snes.ksp
        oldksp.incrementTabLevel(teamno*2)
        defksp = DeflatedKSP(deflation, y, oldksp, snes)
        snes.ksp = PETSc.KSP().createPython(defksp, comm)
        snes.ksp.pc.setType('none')

    try:
        solver.solve()
    except ConvergenceError:
        pass
    except:
        import traceback
        traceback.print_exc()
        pass

    success = snes.getConvergedReason() > 0
    iters   = snes.getIterationNumber()
    return (success, iters)
