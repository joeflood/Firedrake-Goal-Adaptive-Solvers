# -*- coding: utf-8 -*-
import sys
from   math import floor

from defcon import *
from dolfin import *

import matplotlib.pyplot as plt
from numpy import sqrt as nsqrt

class CarrierProblem(BifurcationProblem):
    def __init__(self):
        self.bcs = None

        # Awesome asymptotic formulae from Jon Chapman
        self.pitchbfs = [(0.472537/n) for n in range(1,100)]
        self.foldbfs  = [(0.472537/(n - 0.8344/n)) for n in range(2, 100)]

    def mesh(self, comm):
        return IntervalMesh(comm, 10000, -1, 1)

    def function_space(self, mesh):
        return FunctionSpace(mesh, "CG", 1)

    def parameters(self):
        eps   = Constant(0)
        return [(eps, "epsilon", r"$\varepsilon$")]

    def residual(self, y, params, v):
        eps = params[0]
        x = SpatialCoordinate(y.function_space().mesh())[0]

        F = (
            - eps**2*inner(grad(y), grad(v))*dx
            + 2*(1-x*x) * inner(y, v)*dx
            + inner(y*y, v)*dx
            - inner(Constant(1), v)*dx
            )

        return F

    def boundary_conditions(self, V, params):
        # The boundary conditions are independent of parameters, so only
        # evaluate them once for efficiency.
        if self.bcs is None:
            self.bcs = [DirichletBC(V, 0.0, "on_boundary")]
        return self.bcs

    def functionals(self):
        def signedL2(y, params):
            V = y.function_space()
            j = assemble(y*y*dx)
            g = project(grad(y)[0], V)((-1.0,))
            return j*g

        return [(signedL2, "signedL2", r"$y'({-1}) \|y\|^2$")]

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, V, params, n):
        return interpolate(Constant(1), V)

    def number_solutions(self, params):
        # Here I know the number of solutions for each value of eps.
        # This cheating allows me to calculate the bifurcation diagram
        # much more quickly. This can be disabled without changing the
        # correctness of the calculations.
        eps = params[0]

        if eps > 0.468508: return 2
        if eps > 0.284605: return 4
        if eps > 0.234521: return 6
        if eps > 0.171756: return 8
        if eps > 0.156844: return 10
        if eps > 0.124097: return 12
        if eps > 0.117898: return 14
        if eps > 0.1:      return 16

        # Or alternatively use Jon Chapman's asymptotic formulae:
        nbifurcations = len([x for x in self.pitchbfs + self.foldbfs if x >= eps])
        return (nbifurcations+1)*2

    def squared_norm(self, a, b, params):
        eps = Constant(params[0])

        return inner(a - b, a - b)*dx + sqrt(eps)*inner(grad(a - b), grad(a - b))*dx

    def solver_parameters(self, params, task, **kwargs):
        return {
               "snes_max_it": 100,
               "snes_atol": 1.0e-9,
               "snes_rtol": 0.0,
               "snes_divergence_tolerance": -1,
               "snes_monitor": None,
               "ksp_type": "preonly",
               "pc_type": "lu"
               }

if __name__ == "__main__":
    dc = DeflatedContinuation(problem=CarrierProblem(), teamsize=1, verbose=True)
    epssq = list(linspace(0.25, 0.03, 441)) + list(linspace(0.03, 0.01, 201))[1:]
    eps   = nsqrt(epssq)
    dc.run(values={"epsilon": list(eps)})

    dc.bifurcation_diagram("signedL2")
    ax = plt.gca()
    ax.set_xscale('log')
    plt.ylim([-100, 100])
    plt.xlim([0.01, 0.25])
    plt.title(r"Solutions of the Carrier problem")
    plt.savefig("bifurcation.pdf")
